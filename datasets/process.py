# coding=utf-8
"""
@author:cxy
@file: process.py
@date: 2024/4/2 20:56
"""
from Bio.PDB import PDBParser, MMCIFParser
import torch
import os
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from utils.torsion import get_sidechain_torsion
import numpy as np
from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det
from scipy import spatial
from scipy.special import softmax
from scipy.spatial.transform import Rotation
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from torch_cluster import radius_graph


periodic_table = GetPeriodicTable()

biopython_pdbparser = PDBParser(QUIET=True)
biopython_cifparser = MMCIFParser()

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 14)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def rec_residue_featurizer(rec):
    feature_list = []
    for residue in rec.get_residues():
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def parse_structures(pdbid, pdbbind_dir):
    state1_rec, state2_rec = parsePDB(pdbid, pdbbind_dir)
    return state1_rec, state2_rec


def parsePDB(pdbid, pdbbind_dir):
    file_paths = os.listdir(os.path.join(pdbbind_dir, pdbid))  # 遍历文件夹
    crystal_rec_path = os.path.join(pdbbind_dir, pdbid, [path for path in file_paths if 'state1' in path][0])
    state1_rec_path = os.path.join(pdbbind_dir, pdbid, [path for path in file_paths if 'state2' in path][0])
    return parse_pdb_from_path(crystal_rec_path), parse_pdb_from_path(state1_rec_path)


def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        if path[-4:] == '.pdb':
            structure = biopython_pdbparser.get_structure('pdb', path)
        else:
            raise ValueError("protein is not pdb")
        rec = structure[0]
    return rec


def extract_structure(rec, lm_embedding_chains=None):
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    chis = []
    chi_masks = []
    valid_chain_ids = []
    lengths = []
    torsion_worker = get_sidechain_torsion()
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_chis = []
        chain_chi_masks = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                chi_list, chi_mask = torsion_worker.calculate_torsion(residue)
                chain_chis.append(chi_list)
                chain_chi_masks.append(chi_mask)
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        chis.append(np.array(chain_chis))
        chi_masks.append(np.array(chain_chi_masks))
        if not count == 0: valid_chain_ids.append(chain.get_id())
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_chis = []
    valid_chi_masks = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    print(i, lm_embedding_chains)
                    raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                valid_lm_embeddings.append(lm_embedding_chains[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_chis.append(chis[i])
            valid_chi_masks.append(chi_masks[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    chis = np.concatenate(valid_chis, axis=0)
    chi_masks = np.concatenate(valid_chi_masks, axis=0)
    # print(valid_lm_embeddings)
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert len(chis) == len(c_alpha_coords)
    # print(f"chis {len(chis)}----c_alpha_coords {len(c_alpha_coords)}")
    assert len(chi_masks) == len(c_alpha_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, lm_embeddings


def get_align_rotran(coords, reference_coords):
    # center on centroid
    av1 = coords.mean(0, keepdims=True)
    av2 = reference_coords.mean(0, keepdims=True)
    coords = coords - av1
    reference_coords = reference_coords - av2
    # correlation matrix
    a = dot(transpose(coords), reference_coords)
    u, d, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    # check if we have found a reflection
    if det(rot) < 0:
        vt[2] = -vt[2]
        rot = transpose(dot(transpose(vt), transpose(u)))
    tran = av2 - dot(av1, rot)
    return tran, rot


def get_calpha_graph(name, rec, state1, c_alpha_coords, n_coords, c_coords, chis, chi_masks, structure_graph, cutoff=20, max_neighbor=None, lm_embeddings=None, profile_path=None):

    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph
    distances = spatial.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    lf_3pts = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'{name} : The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
            assert 1==0, 'isolated residue'
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(torch.tensor(diff_vecs))  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
        lf_3pts.append(np.array([n_coords[i], c_alpha_coords[i], c_coords[i]]))

    assert len(src_list) == len(dst_list)
    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))
    structure_graph['stru'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], dim=1) if lm_embeddings is not None else node_feat
    structure_graph['stru'].pos = torch.from_numpy(c_alpha_coords).float()
    structure_graph['stru'].lf_3pts = torch.from_numpy(np.array(lf_3pts)).float()
    # structure_graph['receptor'].local_frames = get_local_frames(torch.from_numpy(np.array(lf_3pts)).float())
    structure_graph['stru'].mu_r_norm = mu_r_norm
    structure_graph['stru'].chis = torch.from_numpy(chis).float()
    structure_graph['stru'].acc_pred_chis = torch.zeros_like(structure_graph['stru'].chis[:, :5]).float()
    structure_graph['stru'].chi_masks = torch.from_numpy(chi_masks[:, :7]).float()
    structure_graph['stru'].chi_symmetry_masks = torch.from_numpy(chi_masks[:, 7:]).long()
    structure_graph['stru'].side_chain_vecs = side_chain_vecs.float()
    structure_graph['stru', 'rec_contact', 'stru'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    # 加载结构谱特征##################################

    profile_npz = np.load(profile_path)
    profile = profile_npz["profile"]  # (L, L, 36)
    entropy = profile_npz["entropy"]  # (L, L, 1)
    combined = np.concatenate([profile, entropy], axis=-1).mean(axis=1)

    # 存为结构图的 profile
    structure_graph['stru'].profile = torch.from_numpy(combined).float()
    # print(profile_path)

    # 加载结构谱特征##################################


    # 给结构中添加另一个态
    if state1 is not None:
        state1_rec, state1_coords, state1_c_alpha_coords, state1_n_coords, state1_c_coords, state1_chis, state1_chi_masks, state1_lm_embeddings = extract_structure(state1)
        assert len(state1_c_alpha_coords) == len(c_alpha_coords), f'{name} ca ne crystal'
        state1_trans = []
        state1_trans_sigma = []
        state1_rotvecs = []
        state1_rotvecs_sigma = []
        for i, ref_coors in enumerate(state1_c_alpha_coords):
            lf_3pts = np.array([n_coords[i], c_alpha_coords[i], c_coords[i]])
            state1_lf_3pts = np.array([state1_n_coords[i], state1_c_alpha_coords[i], state1_c_coords[i]])
            tran, rot = get_align_rotran(lf_3pts - lf_3pts[[1]], state1_lf_3pts - lf_3pts[[1]])
            state1_trans.append(tran)
            state1_trans_sigma.append(np.linalg.norm(tran, axis=-1))
            state1_rotvecs.append(Rotation.from_matrix(rot.T).as_rotvec())
            state1_rotvecs_sigma.append(np.linalg.norm(state1_rotvecs[-1], axis=-1))
        structure_graph['stru'].state1_trans = torch.from_numpy(np.concatenate(state1_trans)).float()
        structure_graph['stru'].state1_trans_sigma = torch.from_numpy(np.concatenate(state1_trans_sigma)).float()
        structure_graph['stru'].state1_rotvecs = torch.from_numpy(np.array(state1_rotvecs)).float()
        structure_graph['stru'].state1_rotvecs_sigma = torch.from_numpy(np.array(state1_rotvecs_sigma)).float()
        structure_graph['stru'].state1_chis = torch.from_numpy(state1_chis - chis).float()
        structure_graph['stru'].chi_masks = torch.from_numpy(chi_masks[:, :7] & state1_chi_masks[:, :7]).float()
    return


def rec_atom_featurizer(rec):
    atom_feats = []
    for i, atom in enumerate(rec.get_atoms()):
        atom_name, element = atom.name, atom.element
        if element == 'CD':
            element = 'C'
        assert not element == ''
        try:
            atomic_num = periodic_table.GetAtomicNumber(element)
        except:
            atomic_num = -1
        atom_feat = [safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                     safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                     safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                     safe_index(allowable_features['possible_atom_type_3'], atom_name)]
        atom_feats.append(atom_feat)
    return atom_feats


def get_rec_graph(name, rec, state1, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, structure_graph, rec_radius, c_alpha_max_neighbors=None, all_atoms=False,
                  atom_radius=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None, profile_path=None):
    if all_atoms:
        return get_fullrec_graph(name, rec, rec_coords, c_alpha_coords, n_coords, c_coords, structure_graph,
                                 c_alpha_cutoff=rec_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
                                 atom_cutoff=atom_radius, atom_max_neighbors=atom_max_neighbors, remove_hs=remove_hs, lm_embeddings=lm_embeddings)
    else:
        return get_calpha_graph(name, rec, state1, c_alpha_coords, n_coords, c_coords, chis, chi_masks, structure_graph, rec_radius, c_alpha_max_neighbors, lm_embeddings=lm_embeddings, profile_path=profile_path)


def get_fullrec_graph(name, rec, rec_coords, c_alpha_coords, n_coords, c_coords, structure_graph, c_alpha_cutoff=20,
                      c_alpha_max_neighbors=None, atom_cutoff=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    # builds the receptor graph with both residues and atoms

    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")
    # Build the k-NN graph of residues
    distances = spatial.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    lf_3pts = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < 0)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'{name}_res{i}: The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
            assert 1==0, 'isolated residue'
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert 1 - 1e-2 < weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
        lf_3pts.append(np.array([n_coords[i], c_alpha_coords[i], c_coords[i]]))
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    structure_graph['stru'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], dim=1) if lm_embeddings is not None else node_feat
    structure_graph['stru'].pos = torch.from_numpy(c_alpha_coords).float()
    structure_graph['stru'].lf_3pts = torch.from_numpy(np.array(lf_3pts)).float()
    structure_graph['stru'].mu_r_norm = mu_r_norm
    structure_graph['stru'].side_chain_vecs = side_chain_vecs.float()
    structure_graph['stru', 'rec_contact', 'stru'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec_coords)])
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
    atom_coords = torch.from_numpy(np.concatenate(rec_coords, axis=0)).float()

    if remove_hs:
        not_hs = (atom_feat[:, 1] != 0)
        src_c_alpha_idx = src_c_alpha_idx[not_hs]
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]

    atoms_edge_index = radius_graph(atom_coords, atom_cutoff, max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
    atom_res_edge_index = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()

    structure_graph['atom'].x = atom_feat
    structure_graph['atom'].pos = atom_coords
    structure_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
    structure_graph['atom', 'atom_rec_contact', 'stru'].edge_index = atom_res_edge_index
    return
