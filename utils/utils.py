# coding=utf-8
"""
@author:cxy
@file: utils.py
@date: 2024/3/27 13:40
"""
import random
import torch
from torch.nn import DataParallel
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
periodic_table = GetPeriodicTable()
from modules.model import ScoreModel
from utils.diffusion_utils import get_timestep_embedding
from torch_geometric.data import Dataset, Data


def split_dataset(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = len(data) - train_size
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]
    

def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    model_class = ScoreModel  # 返回预测的res_tr_pred,res_rot_pred,res_chi_pred
    '''
    embedding type使用方式是sinusoidal
    sinusoidal是一种表示嵌入的类型，嵌入将高维空间中的对象映射到低维空间的方法，通常将离散的、不连续的数据表示为
    低维的，稠密的向量
    '''
    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        scale_by_sigma=args.scale_by_sigma,  # 是否对分数进行归一化
                        sigma_embed_dim=args.sigma_embed_dim,  # 32
                        ns=args.ns, nv=args.nv,  # ns:16, nv:4
                        distance_embed_dim=args.distance_embed_dim,  # 32
                        batch_norm=not args.no_batch_norm,  # 是否使用批量归一化
                        dropout=args.dropout,  # 防止过拟合，默认为0
                        use_second_order_repr=args.use_second_order_repr,
                        lm_embedding_type=lm_embedding_type)  # esm

    # if device.type == 'cuda' and not no_parallel:
    #     model = DataParallel(model) # 用于在多个GPU上并行运行模型，自动将输入数据和模型单数分布到多个GPU上
    model.to(device)
    return model


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    # 学习率调度器，动态调整学习率
    if args.scheduler == 'plateau':  # 默认为这个选项
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5,
                                                               patience=args.scheduler_patience, min_lr=(args.lr*0.1) / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]