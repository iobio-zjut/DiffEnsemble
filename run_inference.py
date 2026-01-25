# coding=utf-8
"""
@author:cxy
@file: run_inference.py
@date: 2024/9/20 10:30
"""
import copy
import os
import warnings
import time
import numpy as np
import scipy
import torch
import random

from datasets.pre_multiCon_x import *
from datasets.process import parse_pdb_from_path
from utils.clash import compute_side_chain_metrics
from utils.visualise import save_protein, modify_pdb
from argparse import ArgumentParser
from functools import partial
from torch_geometric.loader import DataLoader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.utils import read_strings_from_txt
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')


parser = ArgumentParser()
parser.add_argument('--protein_path', type=str, required=True, help='Path to the target .pdb file')
parser.add_argument('--target_txt', type=str, required=True, help='protein name')
parser.add_argument('--out_dir', type=str, default='./target_results', help='Directory where the outputs will be written to')
parser.add_argument('--esm_embeddings_path', type=str, default='esm_folder', help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--profile_features', type=str, default="profile_features", help='test dataset')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--savings_per_complex', type=int, default=10, help='Number of samples to save')
parser.add_argument('--seed', type=int, default=17, help='set seed number')
parser.add_argument('--model_dir', type=str, default='/save_models', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--inference_num', type=int, default=0, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--cache_path', type=str, default='/cache', help='Folder from where to load/restore cached dataset')
parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
parser.add_argument('--inference_steps', type=int, default=10, help='Number of denoising steps')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--keep_local_structures', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
parser.add_argument('--protein_dynamic', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--relax', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--use_existing_cache', action='store_true', default=False, help='Use existing cache file, if they exist.')
parser.add_argument("--num_blocks", "-numb", action="store", type=int, default=8, help="# of reidual blocks (Default: 8)")
parser.add_argument('--sample_batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--resume', type=str, default=False, help='Whether it is necessary to continue running from the breakpoint.')
parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
parser.add_argument('--res_tr_weight', type=float, default=0.2, help='Weight of residue translation loss')
parser.add_argument('--res_rot_weight', type=float, default=1, help='Weight of residue rotation loss')
parser.add_argument('--res_chi_weight', type=float, default=20, help='Weight of residue rotation loss')
parser.add_argument('--rot_sigma_min', type=float, default=0.03, help='Minimum sigma for rotational component')
parser.add_argument('--rot_sigma_max', type=float, default=1.65, help='Maximum sigma for rotational component')
parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
parser.add_argument('--tr_sigma_max', type=float, default=20, help='Maximum sigma for translational component')
parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
parser.add_argument('--res_rot_sigma_min', type=float, default=0.01, help='Minimum sigma for translational component')
parser.add_argument('--res_rot_sigma_max', type=float, default=1, help='Maximum sigma for translational component')
parser.add_argument('--res_tr_sigma_min', type=float, default=0.01, help='Minimum sigma for translational component')
parser.add_argument('--res_tr_sigma_max', type=float, default=1, help='Maximum sigma for translational component')
parser.add_argument('--res_chi_sigma_min', type=float, default=0.01, help='Minimum sigma for translational component')
parser.add_argument('--res_chi_sigma_max', type=float, default=1, help='Maximum sigma for translational component')
parser.add_argument('--structure_radius', type=float, default=30, help='Cutoff on distances for receptor edges')
parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='Maximum number of neighbors for each residue')
parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')
parser.add_argument('--atom_radius', type=float, default=5, help='Cutoff on distances for atom connections')
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
parser.add_argument('--no_torsion', action='store_true', default=False, help='')
parser.add_argument('--num_dataloader_workers', type=int, default=1, help='Number of workers for dataloader')
parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')
# model parameters
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='MLP dropout')
parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
parser.add_argument('--embedding_scale', type=int, default=10000, help='')
parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
parser.add_argument('--only_test', action='store_true', default=False, help='If only test')

parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--w_decay', type=float, default=1e-5, help='')
parser.add_argument('--scheduler', type=str, default='plateau', help='')
parser.add_argument('--scheduler_patience', type=int, default=10, help='')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--finetune_freq', type=int, default=None, help='Frequency of epochs for which to run finetune on train data')
parser.add_argument('--num_finetune_complexes', type=int, default=500, help='Number of complexes for finetune')
parser.add_argument('--val_inference_freq', type=int, default=1000, help='Frequency of epochs for which to run expensive inference on val data')
parser.add_argument('--num_inference_structures', type=int, default=100, help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
parser.add_argument('--test_sigma_intervals', action='store_true', default=False, help='Whether to log loss per noise interval')
parser.add_argument('--use_ema', action='store_true', default=False, help='Whether or not to use ema for the model weights')
#energy
parser.add_argument('--use-charmm36', type=lambda x: eval(x), default=False)
parser.add_argument('--fix-pdb', type=lambda x: eval(x), default=True)
parser.add_argument('--add-H', type=lambda x: eval(x), default=True)
parser.add_argument('--minimize-struct', type=lambda x: eval(x), default=False)
parser.add_argument('--fix-ca-only', type=lambda x: eval(x), default=False)
parser.add_argument('--stiffness', type=float, default=10.)
parser.add_argument('--tolerance', type=float, default=2.39)
parser.add_argument('--max-iter', type=int, default=0)
parser.add_argument('--n-proc', type=int, default=1)
parser.add_argument('--n-threads', type=int, default=None)
parser.add_argument('--use-gpu', type=lambda x: eval(x), default=False)

args = parser.parse_args()


def Seed_everything(seed=17):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


Seed_everything(seed=args.seed)
os.makedirs(args.out_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}.......')

data_list = read_strings_from_txt(args.target_txt)
protein_path_list = []
for i in range(len(data_list)):
    path = os.path.join(args.protein_path, f"Diff_V1/{data_list[i]}.pdb")
    protein_path_list.append(path)
test_dataset = Pre_multiConf(transform=None, cache_path=args.cache_path, data_list=data_list, data_path=protein_path_list,
                         structure_radius=args.structure_radius, num_workers=1, c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                         all_atoms=args.all_atoms, atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors,
                         esm_embeddings_path=args.esm_embeddings_path, use_existing_cache=args.use_existing_cache, profile_features_path=args.profile_features)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
print(len(test_loader))
print("load dataset was completed...........")
t_to_sigma = partial(t_to_sigma_compl, args=args) 

model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(f'{args.model_dir}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
print("loading model completed..... ")
model = model.to(device)
tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
res_tr_schedule = tr_schedule
res_rot_schedule = tr_schedule
res_chi_schedule = tr_schedule
print('common t schedule', tr_schedule)

failures, skipped, confidences_list, names_list, run_times, min_self_distances_list = 0, 0, [], [], [], []
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))
print(f"The process of {args.protein_path}")


def predict_one_complex(orig_structure_graph, model, res_tr_schedule, res_rot_schedule, res_chi_schedule, t_to_sigma, N, args, device, ):
    data_list = [copy.deepcopy(orig_structure_graph)]
    randomize_position(data_list, args.no_torsion, args.no_random, args.res_tr_sigma_max, args.res_rot_sigma_max)
    data_list_randomized = copy.deepcopy(data_list)
    now_pdbpath = os.path.join(args.protein_path, f"Diff_V1/{orig_structure_graph['name'][0]}.pdb")
    receptor_pdb = parse_pdb_from_path(now_pdbpath)  
    start_time = time.time()
    visualization_list = None
    steps = args.inference_steps
    final_data_list, data_list_step = [], [[] for _ in range(steps)]
    for i in range(int(np.ceil(len(data_list)/args.batch_size))):
        outputs = sampling(data_list=data_list[i*args.batch_size:(i+1)*args.batch_size], model=model,
                            inference_steps=steps, res_tr_schedule=res_tr_schedule, res_rot_schedule=res_rot_schedule,
                            res_chi_schedule=res_chi_schedule, device=device, t_to_sigma=t_to_sigma,
                            model_args=args, no_random=args.no_random, ode=args.ode,
                            visualization_list=visualization_list, batch_size=args.batch_size,
                            no_final_step_noise=args.no_final_step_noise, protein_dynamic=args.protein_dynamic)
        final_data_list.extend(outputs[0])
        for si in range(steps):
            data_list_step[si].extend(outputs[1][si])

    run_times.append(time.time() - start_time)
    write_dir = f"{args.out_dir}/Diff_V1_result"
    os.makedirs(write_dir, exist_ok=True)
    pdbFiles = []
    a = copy.deepcopy(receptor_pdb)
    modify_pdb(a, final_data_list[0])
    for rank, order in enumerate(range(args.samples_per_complex)):
        new_receptor_pdb = copy.deepcopy(receptor_pdb)
        modify_pdb(new_receptor_pdb, data_list_step[order][0])
        pdbFile = os.path.join(write_dir, f"{orig_structure_graph['name'][0]}{rank+1}.pdb")
        save_protein(new_receptor_pdb, pdbFile)
        pdbFiles.append(pdbFile)
    names_list = write_dir.split("/")[-2]
    return names_list

for idx, orig_structure_graph in tqdm(enumerate(test_loader)):
    names_list = predict_one_complex(orig_structure_graph, model,
                            res_tr_schedule, res_rot_schedule, res_chi_schedule, t_to_sigma, N, args, device)
print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
