# coding=utf-8
"""
@author:cxy
@file: visualise.py
@date: 2024/4/18 15:17
"""
import torch
from Bio.PDB import PDBIO, MMCIFIO, Select
from utils.affine import T
import numpy as np
from scipy.spatial.transform import Rotation as R


def save_protein(s, proteinFile, ca_only=False):
    # if proteinFile[-3:] == 'pdb':
    io = PDBIO()
    # else:
    #     print("protein is not pdb")

    class MySelect(Select):
        def accept_atom(self, atom):
            if atom.get_name() == 'CA':
                return True
            else:
                return False

    class RemoveHs(Select):
        def accept_atom(self, atom):
            if atom.element != 'H':
                return True
            else:
                return False
    io.set_structure(s)
    if ca_only:
        io.save(proteinFile, MySelect())
    else:
        io.save(proteinFile, RemoveHs())
    return None


# a tuple, left bond atom, right bond atom, list of atoms should rotate with the bond.
chi1_bond_dict = {
    "ALA":None,
    "ARG":("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP":("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS":("CA", "CB", ["SG"]),
    "GLN":("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU":("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU":("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS":("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET":("CA", "CB", ["CG", "SD", "CE"]),
    "PHE":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CA", "CB", ["CG", "CD"]),
    "SER":("CA", "CB", ["OG"]),
    "THR":("CA", "CB", ["CG2", "OG1"]),
    "TRP":("CA", "CB", ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":("CA", "CB", ["CG1", "CG2"])
}

chi2_bond_dict = {
    "ALA":None,
    "ARG":("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CB", "CG", ["ND2", "OD1"]),
    "ASP":("CB", "CG", ["OD1", "OD2"]),
    "CYS":None,
    "GLN":("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU":("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CB", "CG1", ["CD1"]),
    "LEU":("CB", "CG", ["CD1", "CD2"]),
    "LYS":("CB", "CG", ["CD", "CE", "NZ"]),
    "MET":("CB", "CG", ["SD", "CE"]),
    "PHE":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CB", "CG", ["CD"]),
    "SER":None,
    "THR":None,
    "TRP":("CB", "CG", ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":None,
}


chi3_bond_dict = {
    "ALA":None,
    "ARG":("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN":None,
    "ASP":None,
    "CYS":None,
    "GLN":("CG", "CD", ["NE2", "OE1"]),
    "GLU":("CG", "CD", ["OE1", "OE2"]),
    "GLY":None,
    "HIS":None,
    "ILE":None,
    "LEU":None,
    "LYS":("CG", "CD", ["CE", "NZ"]),
    "MET":("CG", "SD", ["CE"]),
    "PHE":None,
    "PRO":None,
    "SER":None,
    "THR":None,
    "TRP":None,
    "TYR":None,
    "VAL":None,
}

chi4_bond_dict = {
    "ARG":("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS":("CD", "CE", ["NZ"]),

}

chi5_bond_dict = {
    "ARG":("NE", "CZ", ["NH1", "NH2"]),
}


def rotate_chi(res, pred_chi, chi_mask):
    resname = res.resname  # 氨基酸名称 ILE
    for i in range(5):
        if chi_mask[i] == 0:
            continue
        chi_bond_dict = eval(f'chi{i+1}_bond_dict')

        atom1, atom2, rotate_atom_list = chi_bond_dict[resname]
        eps = 1e-6
        if (atom1 not in res) or (atom2 not in res):
            continue
        atom1_coord = res[atom1].coord
        atom2_coord = res[atom2].coord
        rot_vec = atom2_coord - atom1_coord
        rot_vec = pred_chi[i] * (rot_vec) / (np.linalg.norm(rot_vec) + eps)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        for rotate_atom in rotate_atom_list:
            if rotate_atom not in res:
                continue
            new_coord = np.matmul(res[rotate_atom].coord - res[atom1].coord, rot_mat.T) + res[atom1].coord
            res[rotate_atom].set_coord(new_coord)
    return res


def modify_pdb(ppdb, data):
    pred_chis, chi_masks = data['stru'].acc_pred_chis.cpu().numpy(), data['stru'].chi_masks.cpu().numpy()
    chi_masks = chi_masks[:, [0, 2, 4, 5, 6]]
    new_df = []
    i = 0
    pred_lf = T.from_3_points(p_xy_plane=data['stru'].lf_3pts[:, 0, :], origin=data['stru'].lf_3pts[:, 1, :], p_neg_x_axis=data['stru'].lf_3pts[:, 2, :])
    all_res = list(ppdb.get_residues())
    for res_idx, res in enumerate(all_res):
        if res.resname == 'HOH':
            continue
        if 'CA' not in res or 'N' not in res or 'C' not in res:
            continue
        c_alpha = torch.tensor(res['CA'].coord).float().unsqueeze(0)
        n = torch.tensor(res['N'].coord).float().unsqueeze(0)
        c = torch.tensor(res['C'].coord).float().unsqueeze(0)
        all_atom = torch.tensor(np.stack([atom.coord for atom in res.get_atoms()])).float()
        lf = T.from_3_points(p_xy_plane=n, origin=c_alpha, p_neg_x_axis=c)
        lf_all_atom = lf.invert_apply(all_atom)
        pred_all_atom = pred_lf[res_idx].apply(lf_all_atom) + data.original_center
        i = 0
        for atom in res.get_atoms():
            atom.set_coord(pred_all_atom[i])
            i += 1
        res = rotate_chi(res, pred_chis[res_idx], chi_masks[res_idx])
    return ppdb