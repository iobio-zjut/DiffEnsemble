# coding=utf-8
"""
@author:cxy
@file: diffusion_utils.py
@date: 2024/3/26 15:38
"""
import torch
from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles
import numpy as np
from torch import nn
import math
import torch.nn.functional as F


def t_to_sigma(t_res_tr, t_res_rot, t_res_chi, args):
    if torch.is_tensor(t_res_tr):
        res_tr_sigma = torch.clamp(args.res_tr_sigma_min + (args.res_tr_sigma_max-args.res_tr_sigma_min) * (t_res_tr*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_tr.device)) #** (1-t_res_tr) * args.res_tr_sigma_max ** t_res_tr
        res_rot_sigma = torch.clamp(args.res_rot_sigma_min + (args.res_rot_sigma_max-args.res_rot_sigma_min) * (t_res_rot*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_rot.device)) #** (1-t_res_rot) * args.res_rot_sigma_max ** t_res_rot
        res_chi_sigma = torch.clamp(args.res_chi_sigma_min + (args.res_chi_sigma_max-args.res_chi_sigma_min) * (t_res_chi*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_chi.device))
    else:
        res_tr_sigma = min(args.res_tr_sigma_min + (args.res_tr_sigma_max-args.res_tr_sigma_min) * (t_res_tr*5.) ** 0.3, 1.) #** (1-t_res_tr) * args.res_tr_sigma_max ** t_res_tr
        res_rot_sigma = min(args.res_rot_sigma_min + (args.res_rot_sigma_max-args.res_rot_sigma_min) * (t_res_rot*5.) ** 0.3, 1.) #** (1-t_res_rot) * args.res_rot_sigma_max ** t_res_rot
        res_chi_sigma = min(args.res_chi_sigma_min + (args.res_chi_sigma_max-args.res_chi_sigma_min) * (t_res_chi*5.) ** 0.3, 1.)
    return res_tr_sigma, res_rot_sigma, res_chi_sigma



def modify_conformer(data, res_tr_update, res_rot_update, res_chi_update):
    res_rot_mat = axis_angle_to_matrix(res_rot_update)
    data['stru'].lf_3pts = (data['stru'].lf_3pts - data['stru'].lf_3pts[:, [1], :]) @ res_rot_mat.transpose(1, 2) + data['stru'].lf_3pts[:, [1], :] + res_tr_update[:, None, :]
    data['stru'].pos = data['stru'].pos + res_tr_update
    if 'acc_pred_chis' in data['stru'].keys():
        data['stru'].acc_pred_chis = ((data['stru'].acc_pred_chis + res_chi_update) * data['stru'].chi_masks[:, [0, 2, 4, 5, 6]]) % (2*np.pi)
    res_chi_update = res_chi_update[:, [0, 0, 1, 1, 2, 3, 4]]
    data['stru'].chis = ((data['stru'].chis + res_chi_update) * data['stru'].chi_masks) % (2*np.pi)
    min_chi1 = torch.minimum(data['stru'].chis[:, 0], data['stru'].chis[:, 1])
    max_chi1 = torch.maximum(data['stru'].chis[:, 0], data['stru'].chis[:, 1])
    data['stru'].chis[:, 0] = max_chi1
    data['stru'].chis[:, 1] = min_chi1
    min_chi2 = torch.minimum(data['stru'].chis[:, 2], data['stru'].chis[:, 3])
    max_chi2 = torch.maximum(data['stru'].chis[:, 2], data['stru'].chis[:, 3])
    data['stru'].chis[:, 2] = max_chi2
    data['stru'].chis[:, 3] = min_chi2
    return data


def set_time(complex_graphs, t_res_tr, t_res_rot, t_res_chi, batchsize, all_atoms, device):
    complex_graphs['stru'].node_t = {
        'tr': t_res_tr * torch.ones(complex_graphs['stru'].num_nodes).to(device),
        'rot': t_res_rot * torch.ones(complex_graphs['stru'].num_nodes).to(device),
        'tor': t_res_chi * torch.ones(complex_graphs['stru'].num_nodes).to(device)}

    complex_graphs.complex_t = {'res_tr': t_res_tr * torch.ones(complex_graphs['stru'].num_nodes).to(device),
                                'res_rot': t_res_rot * torch.ones(complex_graphs['stru'].num_nodes).to(device),
                                'res_chi': t_res_chi * torch.ones(complex_graphs['stru'].num_nodes).to(device)}

    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_res_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_res_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_res_chi * torch.ones(complex_graphs['atom'].num_nodes).to(device)}



def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb



class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb
    

def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]
