# coding=utf-8
"""
@author:cxy
@file: torsion.py
@date: 2024/3/26 18:58
"""
import numpy as np
import math
from Bio import PDB
import torch, copy
from scipy.spatial.transform import Rotation as R


class get_sidechain_torsion(object):
    """
    Calculate side-chain torsion angles (also known as dihedral or chi angles).
    Depends: Biopython (http://www.biopython.org)
    """
    chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        altchi1=dict(
            VAL=['N', 'CA', 'CB', 'CG2'],
        ),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        altchi2=dict(
            ASP=['CA', 'CB', 'CG', 'OD2'],
            LEU=['CA', 'CB', 'CG', 'CD2'],
            PHE=['CA', 'CB', 'CG', 'CD2'],
            TYR=['CA', 'CB', 'CG', 'CD2'],
        ),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
    )

    default_chi = [1,2,3,4,5]
    # default_outfile = "torsion_list.csv"

    def __init__(self,chi=None, units="radians"):
        """Set parameters and calculate torsion values"""
        # Configure chi
        if chi is None:
            chi = self.default_chi
        chi_names = list()
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in self.chi_atoms.keys():
                chi_names.append(reg_chi)
                alt_chi = "altchi%s" % x
                if alt_chi in self.chi_atoms.keys():
                    chi_names.append(alt_chi)
            # else:
            #     logging.warning("Invalid chi %s", x)
        # print(chi_names)
        self.chi_names = chi_names
        self.fieldnames = ["id", "model", "chain", "resn", "resi"] + self.chi_names
        # logging.debug("Calculating chi angles: %s", ", ".join(chi_names))

        # Configure units (degrees or radians)
        if units is None:
            units = "degrees"
        self.degrees = bool(units[0].lower() == "d")
        if self.degrees:
            message = "Using degrees"
        else:
            message = "Using radians"
        # logging.debug(message)

    def calculate_torsion(self, res):
        """Calculate side-chain torsion angles for given res"""
        # Skip heteroatoms
        # print("computing")
        for _ in range(1):
            chi_list = [0] * len(self.chi_names)
            mask = [0] * len(self.chi_names)
            symmetry_mask = [0] * 5
            if res.id[0] != " ":
                # print("skip heteroatoms")
                continue
            res_name = res.resname
            if res_name in ("ALA", "GLY"):
                # print("skip ALA and GLY")
                continue
            for x, chi in enumerate(self.chi_names):
                chi_res = self.chi_atoms[chi]
                try:
                    atom_list = chi_res[res_name]
                except KeyError:
                    continue
                try:
                    vec_atoms = [res[a] for a in atom_list]
                except KeyError:
                    continue
                try:
                    vectors = [a.get_vector() for a in vec_atoms]
                    angle = PDB.calc_dihedral(*vectors)
                except:
                    continue
                angle = angle %(2*np.pi)
                if self.degrees:
                    angle = math.degrees(angle)
                chi_list[x] = angle
                mask[x] = 1

            if res_name in self.chi_atoms['altchi1']:
                if mask[0] == 1 and mask[1] == 1:
                    min_angle = min(chi_list[0], chi_list[1])
                    max_angle = max(chi_list[0], chi_list[1])
                else:
                    min_angle = 0
                    max_angle = 0
                    mask[0] = mask[1] = 0
                chi_list[0] = max_angle
                chi_list[1] = min_angle
            if res_name in self.chi_atoms['altchi2']:
                if mask[2] == 1 and mask[3] == 1:
                    min_angle = min(chi_list[2], chi_list[3])
                    max_angle = max(chi_list[2], chi_list[3])
                else:
                    min_angle = 0
                    max_angle = 0
                    mask[2] = mask[3] = 0
                chi_list[2] = max_angle
                chi_list[3] = min_angle
                if res_name != 'LEU':
                    symmetry_mask[1] = 1

        return chi_list, mask+symmetry_mask


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


class get_sidechain_torsion(object):
    """
    Calculate side-chain torsion angles (also known as dihedral or chi angles).
    Depends: Biopython (http://www.biopython.org)
    """

    chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        altchi1=dict(
            VAL=['N', 'CA', 'CB', 'CG2'],
        ),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        altchi2=dict(
            ASP=['CA', 'CB', 'CG', 'OD2'],
            LEU=['CA', 'CB', 'CG', 'CD2'],
            PHE=['CA', 'CB', 'CG', 'CD2'],
            TYR=['CA', 'CB', 'CG', 'CD2'],
        ),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
    )

    default_chi = [1,2,3,4,5]
    # default_outfile = "torsion_list.csv"

    def __init__(self,chi=None, units="radians"):
        """Set parameters and calculate torsion values"""
        # Configure chi
        if chi is None:
            chi = self.default_chi
        chi_names = list()
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in self.chi_atoms.keys():
                chi_names.append(reg_chi)
                alt_chi = "altchi%s" % x
                if alt_chi in self.chi_atoms.keys():
                    chi_names.append(alt_chi)
            # else:
            #     logging.warning("Invalid chi %s", x)
        # print(chi_names)
        self.chi_names = chi_names
        self.fieldnames = ["id", "model", "chain", "resn", "resi"] + self.chi_names
        # logging.debug("Calculating chi angles: %s", ", ".join(chi_names))

        # Configure units (degrees or radians)
        if units is None:
            units = "degrees"
        self.degrees = bool(units[0].lower() == "d")
        if self.degrees:
            message = "Using degrees"
        else:
            message = "Using radians"
        # logging.debug(message)

    def calculate_torsion(self, res):
        """Calculate side-chain torsion angles for given res"""
        # Skip heteroatoms
        # print("computing")
        for _ in range(1):
            chi_list = [0] * len(self.chi_names)
            mask = [0] * len(self.chi_names)
            symmetry_mask = [0] * 5
            if res.id[0] != " ":
                # print("skip heteroatoms")
                continue
            res_name = res.resname
            if res_name in ("ALA", "GLY"):
                # print("skip ALA and GLY")
                continue
            for x, chi in enumerate(self.chi_names):
                chi_res = self.chi_atoms[chi]
                try:
                    atom_list = chi_res[res_name]
                except KeyError:
                    continue
                try:
                    vec_atoms = [res[a] for a in atom_list]
                except KeyError:
                    continue
                try:
                    vectors = [a.get_vector() for a in vec_atoms]
                    angle = PDB.calc_dihedral(*vectors)
                except:
                    continue
                angle = angle %(2*np.pi)
                if self.degrees:
                    angle = math.degrees(angle)
                chi_list[x] = angle
                mask[x] = 1

            if res_name in self.chi_atoms['altchi1']:
                if mask[0] == 1 and mask[1] == 1:
                    min_angle = min(chi_list[0], chi_list[1])
                    max_angle = max(chi_list[0], chi_list[1])
                else:
                    min_angle = 0
                    max_angle = 0
                    mask[0] = mask[1] = 0
                chi_list[0] = max_angle
                chi_list[1] = min_angle
            if res_name in self.chi_atoms['altchi2']:
                if mask[2] == 1 and mask[3] == 1:
                    min_angle = min(chi_list[2], chi_list[3])
                    max_angle = max(chi_list[2], chi_list[3])
                else:
                    min_angle = 0
                    max_angle = 0
                    mask[2] = mask[3] = 0
                chi_list[2] = max_angle
                chi_list[3] = min_angle
                if res_name != 'LEU':
                    symmetry_mask[1] = 1

        return chi_list, mask+symmetry_mask


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy:pos = torch.from_numpy(pos.astype(np.float32))
    return pos
