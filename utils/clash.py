# coding=utf-8
"""
@author:cxy
@file: clash.py
@date: 2024/4/19 11:24
"""

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from scipy.spatial.distance import cdist


def compute_clash_score(dis, base_vdw_dis, neighbor_mask=None, clash_thr=4):
    mask = dis < clash_thr
    if neighbor_mask is not None:
        mask = mask & neighbor_mask
    n = mask.sum()
    overlap = base_vdw_dis[mask] - dis[mask]
    has_clash = overlap > 0
    clashScore = np.sqrt((overlap[has_clash]**2).sum() / (1e-8 + n))
    return clashScore, overlap[has_clash], has_clash.sum(), n


def compute_side_chain_metrics(pdbFile, vdw_radii_table=None, verbose=True):

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure(pdbFile, pdbFile)  # 获取蛋白质结构
    # compute clash.
    all_atoms = list(s.get_atoms())
    all_heavy_atoms = [atom for atom in all_atoms if atom.element != 'H']
    atom_coords = np.array([atom.coord for atom in all_heavy_atoms])  # 蛋白质坐标

    p_atoms_vdw = np.array([vdw_radii_table[a.element] for a in all_heavy_atoms])  # 蛋白质
    dis = cdist(atom_coords, atom_coords)

    # 这里是判断自己的情况
    base_vdw_dis = p_atoms_vdw.reshape(-1, 1) + p_atoms_vdw.reshape(1, -1)

    clashScore, overlap, clash_n, n = compute_clash_score(dis, base_vdw_dis, clash_thr=4)
    # 控制程序输出信息的详细程度
    if verbose:
        return clashScore, overlap, clash_n, n
    return clashScore