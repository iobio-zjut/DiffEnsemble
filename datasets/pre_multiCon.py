# coding=utf-8

import glob
import pickle
from collections import defaultdict
from multiprocessing import Pool
from random import random
import numpy as np
import pandas as pd
import torch
import os
import copy
from tqdm import tqdm
from Bio.PDB import PDBParser
from datasets.process import get_rec_graph, extract_structure, parse_structures
from utils.diffusion_utils import set_time, modify_conformer
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from utils.utils import *
from datasets.process import parse_pdb_from_path
from datetime import datetime
import subprocess
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform()
        t_res_tr, t_res_tor, t_res_chi = t, t, t
        return self.apply_noise(data, t_res_tr, t_res_tor, t_res_chi)

    def apply_noise(self, data, t_res_tr, t_res_rot, t_res_chi, res_chi_update=None):
        print("using noise......................")
        res_tr_sigma, res_rot_sigma, res_chi_sigma = self.t_to_sigma(t_res_tr, t_res_rot, t_res_chi)
        set_time(data, t_res_tr, t_res_rot, t_res_chi, 1, self.all_atom, device=None)

        res_sigma = torch.clamp(torch.normal(mean=0., std=0.2, size=(1,)).float(), min=0., max=1.)[0]  # 正态分布


        try:
            res_tr_update = data['stru'].state1_trans * res_sigma
            res_rot_update = data['stru'].state1_rotvecs * res_sigma
            res_chi_update = (data['stru'].state1_chis[:, [0, 2, 4, 5, 6]] * res_sigma + torch.normal(mean=0, std=0.3, size=(data['stru'].pos.shape[0], 5))) * data['stru'].chi_masks[:, [0, 2, 4, 5, 6]] if res_chi_update is None else res_chi_update

            modify_conformer(data, res_tr_update, res_rot_update, res_chi_update)
            data.res_tr_score = -res_tr_update
            data.res_rot_score = -res_rot_update  # torch.from_numpy(so3.score_res_vec(vec=res_rot_update, eps=res_rot_sigma.cpu().numpy())).float()
            data.res_chi_score = -res_chi_update
            data.res_loss_weight = torch.ones(data['stru'].pos.shape[0], 1).float()
        except Exception as e:
            print(data['name'] + "apply_noise has error")
            print(e)
        return data


class Pre_multiConf(Dataset):
    def __init__(self, transform=None, cache_path='test_data/cache', data_list=None, data_path=None,
                 structure_radius=30, num_workers=2, c_alpha_max_neighbors=None,
                 keep_original=False, remove_hs=False, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None,
                 use_existing_cache=True):

        super(Pre_multiConf, self).__init__(data_path, transform)
        # self.parallel_count = 200
        self.parallel_count = 2
        self.structure_radius = structure_radius
        self.num_workers = num_workers
        self.data_path = data_path
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.data_list = data_list
        self.cache_path = cache_path
        if all_atoms:
            self.cache_path += '_allatoms'
        current_time = datetime.now().strftime('%m%d_%H%M%S')
        self.full_cache_path = os.path.join(self.cache_path, current_time)
        self.keep_original = keep_original
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if (not use_existing_cache) or (not os.path.exists(os.path.join(self.full_cache_path, "stru_graphs.pkl"))):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.inference_preprocessing()
        # print('loading data from memory: ', os.path.join(self.full_cache_path, "stru_graphs.pkl"))
        with open(os.path.join(self.full_cache_path, "stru_graphs.pkl"), 'rb') as f:
            self.stru_graphs = pickle.load(f)

    def len(self):
        return len(self.stru_graphs)

    def get(self, idx):
        stru_graph = copy.deepcopy(self.stru_graphs[idx])
        return stru_graph

    def process_one_batch(self, param):
        stru_names, lm_embeddings, i = param
        if isinstance(lm_embeddings, str):
            with open(lm_embeddings, 'rb') as f:
                lm_embeddings = pickle.load(f)
        stru_graphs = []
        for idx in tqdm(range(len(stru_names))):
            t = self.get_structure((stru_names[idx], self.data_path, lm_embeddings[idx]))
            stru_graphs.extend(t[0])

        with open(os.path.join(self.full_cache_path, f"stru_graphs{i}.pkl"), 'wb') as f:
            pickle.dump((stru_graphs), f)
    def inference_preprocessing(self):
        structure_list = []
        for idx in range(len(self.data_path)):
            structure_pdb = PDBParser(QUIET=True).get_structure('pdb', self.data_path[idx])
            structure_list.append(structure_pdb)
        lm_embeddings_all = []
        if self.esm_embeddings_path is not None:
            print('Reading language model embeddings.')
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ', self.esm_embeddings_path)
            for protein_path in self.data_path:
                name = protein_path.split("/")[-3]
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, name + '.pt')))
                lm_embeddings = []
                for embeddings_path in embeddings_paths:
                    # print(torch.load(embeddings_path)['representations'][33].shape)  # (L,1280)
                    lm_embeddings.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_all.append(lm_embeddings)
        else:
            lm_embeddings_all = [None] * len(self.data_path)
        print('Generating graphs for proteins')
        if self.num_workers > 1:
            for i in range(len(self.data_path)//self.parallel_count+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"stru_graphs{i}.pkl")):
                    continue
                protein_paths_chunk = self.data_path[self.parallel_count*i:self.parallel_count*(i+1)]
                lm_embeddings = lm_embeddings_all[self.parallel_count*i:self.parallel_count*(i+1)]
                stru_graphs = []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                with tqdm(total=len(protein_paths_chunk), desc=f'loading structures {i}/{len(protein_paths_chunk)//self.parallel_count+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_structure, zip(protein_paths_chunk, lm_embeddings)):
                        stru_graphs.extend(t[0])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)
                with open(os.path.join(self.full_cache_path, f"stru_graphs{i}.pkl"), 'wb') as f:
                    pickle.dump((stru_graphs), f)  # 将信息进行保存
            stru_graphs_all = []
            for i in range(len(self.data_path)//self.parallel_count+1):
                with open(os.path.join(self.full_cache_path, f"stru_graphs{i}.pkl"), 'rb') as f:
                    l = pickle.load(f)
                    stru_graphs_all.extend(l)
            with open(os.path.join(self.full_cache_path, f"stru_graphs.pkl"), 'wb') as f:
                pickle.dump((stru_graphs_all), f)
        else:
            stru_graphs = []
            with tqdm(total=len(self.data_path), desc='loading structures') as pbar:
                for t in map(self.get_structure, zip(self.data_list, self.data_path, lm_embeddings_all)):
                    stru_graphs.extend(t[0])
                    pbar.update()

            with open(os.path.join(self.full_cache_path, "stru_graphs.pkl"), 'wb') as f:
                pickle.dump((stru_graphs), f)

    def get_structure(self, par):  # par:多个参数的元组
        name, protein_path, lm_embedding_chains = par
        state1 = parse_pdb_from_path(f"{protein_path}")

        structure_graph = HeteroData()
        structure_graph.name = name
        structure_graphs = []
        try:
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, lm_embeddings = extract_structure(
                copy.deepcopy(state1), lm_embedding_chains=lm_embedding_chains)

            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                print(f'LM embeddings for {name} did not have the right length for the protein. Skipping {name}.')
            get_rec_graph(name, rec, state1, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, structure_graph, rec_radius=self.structure_radius,
                       c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                         atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)

        try:
            protein_center = torch.mean(structure_graph['stru'].pos, dim=0, keepdim=True)  # 受体中心
            structure_graph['stru'].pos -= protein_center
            structure_graph['stru'].lf_3pts -= protein_center[None, ...]

            if self.all_atoms:
                structure_graph['stru'].pos -= protein_center
            structure_graph.original_center = protein_center
            structure_graphs.append(structure_graph)
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
        return structure_graphs, None
