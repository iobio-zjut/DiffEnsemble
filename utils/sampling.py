# coding=utf-8
"""
@author:cxy
@file: sampling.py
@date: 2024/4/4 10:28
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import set_time, modify_conformer
from utils.geometry import axis_angle_to_matrix
from utils.torsion import modify_conformer_torsion_angles

'''
原本是成批次进行推断，将目标target,10份进行推断，猜测是应为在后期调整的时候一个批次对target进行不同的调整
'''


def randomize_position(data_list, no_torsion, no_random, res_tr_sigma_max, res_rot_sigma_max):

    for structure_graph in data_list:
        if 'state1_trans' in structure_graph['stru'].keys():
            res_tr_update = structure_graph['stru'].state1_trans
            res_rot_update = structure_graph['stru'].state1_rotvecs
            res_chi_update = structure_graph['stru'].state1_chis
            res_rot_mat = axis_angle_to_matrix(res_rot_update)
            structure_graph['stru'].lf_3pts = (structure_graph['stru'].lf_3pts - structure_graph['stru'].lf_3pts[:, [1], :]) @ res_rot_mat.transpose(1, 2) + structure_graph['stru'].lf_3pts[:, [1], :] + res_tr_update[:, None, :]
            structure_graph['stru'].pos = structure_graph['stru'].pos + res_tr_update
            structure_graph['stru'].chis = ((structure_graph['stru'].chis + res_chi_update) * structure_graph['stru'].chi_masks) % (2*np.pi)
        else:
            structure_graph['stru'].chis = (structure_graph['stru'].chis * structure_graph['stru'].chi_masks) % (2*np.pi)

        min_chi1 = torch.minimum(structure_graph['stru'].chis[:, 0], structure_graph['stru'].chis[:, 1])
        max_chi1 = torch.maximum(structure_graph['stru'].chis[:, 0], structure_graph['stru'].chis[:, 1])
        structure_graph['stru'].chis[:, 0] = max_chi1
        structure_graph['stru'].chis[:, 1] = min_chi1
        min_chi2 = torch.minimum(structure_graph['stru'].chis[:, 2], structure_graph['stru'].chis[:,3])
        max_chi2 = torch.maximum(structure_graph['stru'].chis[:, 2], structure_graph['stru'].chis[:,3])
        structure_graph['stru'].chis[:, 2] = max_chi2
        structure_graph['stru'].chis[:, 3] = min_chi2


def sampling(data_list, model, inference_steps, res_tr_schedule, res_rot_schedule, res_chi_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=True, visualization_list=None, confidence_model=None, batch_size=1, no_final_step_noise=False, return_per_step=False, protein_dynamic=True):
    # N = len(data_list)  # 10
    # print(data_list) # HeteroDataBatch对象
    # print(type(data_list))  # list
    data_list_step = []
    '''
    10个target,每个target执行20步
    '''
    for t_idx in range(inference_steps):  # 20步
        # 迭代递推每一个时间步
        t_res_tr, t_res_rot, t_res_chi = res_tr_schedule[t_idx], res_rot_schedule[t_idx], res_chi_schedule[t_idx]
        dt_res_tr = res_tr_schedule[t_idx] - res_tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else res_tr_schedule[t_idx]
        dt_res_rot = res_rot_schedule[t_idx] - res_rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else res_rot_schedule[t_idx]
        dt_res_chi = res_chi_schedule[t_idx] - res_chi_schedule[t_idx + 1] if t_idx < inference_steps - 1 else res_chi_schedule[t_idx]

        # 数据加载
        loader = DataLoader(data_list, batch_size=batch_size)

        for structure_graph_batch in loader:  # 相同的10个目标，一个批次的目标
            b = structure_graph_batch.num_graphs  # batch  10
            n = structure_graph_batch['stru'].pos.shape[0]  # 序列长度 L * 批次数量  1a07这个例子中这里是104 * 10 = 1040
            structure_graph_batch = structure_graph_batch.to(device)

            res_tr_sigma, res_rot_sigma, res_chi_sigma = t_to_sigma(t_res_tr, t_res_rot, t_res_chi)
            set_time(structure_graph_batch, t_res_tr, t_res_rot, t_res_chi, b, model_args.all_atoms, device)

            with torch.no_grad():
                # 预测的xt
                res_tr_score, res_rot_score, res_chi_score = model(structure_graph_batch)

            '''
            上面model得到xt之后，下方进行旋转平移噪声获取
            '''
            res_tr_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_tr_sigma_max / model_args.res_tr_sigma_min)))
            res_rot_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_rot_sigma_max / model_args.res_rot_sigma_min)))
            if ode or 1:
                if protein_dynamic:
                    res_tr_perturb = res_tr_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    res_rot_perturb = res_rot_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                else:
                    res_tr_perturb = torch.zeros((n, 3))
                    res_rot_perturb = torch.zeros((n, 3))
                    res_chi_perturb = torch.zeros((n, 5))
            else:
                res_tr_perturb = torch.zeros((n, 3))
                res_rot_perturb = torch.zeros((n, 3))
                res_chi_z = torch.zeros((n, 5)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(n, 5))
                res_chi_perturb = res_chi_score.cpu() + res_chi_z * dt_res_chi

            res_tr_perturb = torch.clamp(res_tr_perturb, min=-20, max=20)       # safe perturb
            res_per_molecule = res_tr_perturb.shape[0] // b
            # Apply denoise  去噪过程
            # modify_comformer函数：根据传入参数对复合物构象位置角度进行更改,根据给定的更新信息对复合物的构象进行调整，以实现去噪的效果。
            new_data_list = [modify_conformer(structure_graph_batch,
                                          res_tr_perturb[:structure_graph_batch['stru'].pos.shape[0]],
                                          res_rot_perturb[:structure_graph_batch['stru'].pos.shape[0]],
                                          res_chi_perturb[:structure_graph_batch['stru'].pos.shape[0]])]

            data_list = new_data_list
            data_list_step.append(new_data_list)
    return data_list, data_list_step
