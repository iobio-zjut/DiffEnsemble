# coding=utf-8
"""
@author:cxy
@file: model.py
@date: 2024/4/2 15:46
"""
import torch
from torch import nn
from e3nn import o3
from torch.nn import functional as F
from e3nn.nn import BatchNorm
from torch_scatter import scatter, scatter_mean
from torch_cluster import radius, radius_graph
from datasets.process import rec_residue_feature_dims


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.to(self.offset.device).view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
        '''
        first element of feature_dims tuple is a list with the lenght of each categorical
        feature and the second is the number of scalar features
        '''
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        x = x.to("cuda:0")
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], dim=1))
        return x_embedding


class ConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(ConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        # 创建一个张量积网络层，用于处理具有对称性质的输入数据
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        out_nodes = out_nodes or node_attr.shape[0]
        # 根据edge_src张量中的索引位置，在out张量的指定维度上执行scatter操作，将输入张量tp中的值进行填充或累加，并将结果保存在out张量中
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            # print("residual")
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded
        if self.batch_norm:
            # print("batch_norm")
            out = self.batch_norm(out)
        return out


class ScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, rec_max_radius=15, c_alpha_max_neighbors=24,
                 center_max_distance=30, distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dropout=0.0, lm_embedding_type=None, finetune=False):
        super(ScoreModel, self).__init__()
        self.finetune = finetune
        self.t_to_sigma = t_to_sigma
        self.sigma_embed_dim = sigma_embed_dim
        self.rec_max_radius = rec_max_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.num_conv_layers = num_conv_layers

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

        '''
        ns表示空间维度，用于描述旋转操作的表示
        nv旋转轴的维度，通常为3，用于描述三维空间的旋转
        irrep_seq是一个列表，包含了不同阶数的不可约表示
        x0e：表示该不可约表示的特征，x0e表示该表示中只包含偶次项，不包含奇次项
        '''
        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',  # 表示只包含偶次项的一阶不可约表示
                f'{ns}x0e + {nv}x1o + {nv}x2e',  # 表示一阶不可约表示和二阶不可表示的奇次项的组合
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',  # 表示一阶和二阶不可约表示的全部特征的组合
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'  # 表示包含了更高阶的不可约表示的全部特征的组合
            ]
            '''
            odd number奇数项
            even 偶数
            3x0e:3 表示阶数，即球谐函数的阶数为三阶。x0e 表示球谐函数的性质，其中：x 表示球谐函数。0e 表示零阶球谐函数的偶次项
            3x1o:3 表示阶数，即球谐函数的阶数为三阶。x1o 表示球谐函数的性质，其中：x 表示球谐函数。1o 表示一阶球谐函数的奇次项
            3x1e:3 表示阶数，即球谐函数的阶数为三阶。x1e 表示球谐函数的性质，其中：x 表示球谐函数。1e 表示一阶球谐函数的偶次项。
            '''
        else:
            irrep_seq = [
                f'{ns}x0e',  # 表示只包含偶次项的一阶不可约表示
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        rec_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,  # 输入张量的不可约表示
                'sh_irreps': self.sh_irreps,  # 球谐函数的不可约表示
                'out_irreps': out_irreps,  # 输出张量的不可约表示
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            if i == 0:
                parameters['in_irreps'] = parameters['in_irreps'] + ' + 2x1o'

            rec_layer = ConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)

        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)

        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )

        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_chi_final_layer = nn.Sequential(
            nn.Linear(2*ns, ns, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(ns, 5, bias=False)
        )
        if not no_torsion:
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")

            self.tor_final_layer = nn.Sequential(
                nn.Linear(2*ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )

    def forward(self, data, eps=1e-12):
        # build structure graph
        # 传入的加噪数据经过一系列网络层进行预测新的结构
        '''
        # print(rec_node_attr.shape) # [L,1327]
        # print(rec_edge_index.shape) # [2, x1]
        # print(rec_edge_sh.shape) # [x1, 9]
        '''
        # 构图
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        for l in range(len(self.rec_conv_layers)):  # 2层
            # print("rec_node_attr: ")
            # print(rec_node_attr.shape)  # (1144,16) or (1108,16)
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
            # print("rec_edge_attr_: ")
            # print(rec_edge_attr_.shape)  # (26386,48)

            if l == 0:
                # lf_3pts是关键点位置信息 骨架原子信息
                n_vec = data['stru'].lf_3pts[:, 0] - data['stru'].lf_3pts[:, 1]
                n_norm_vec = n_vec / (n_vec.norm(dim=-1, keepdim=True) + eps)
                c_vec = data['stru'].lf_3pts[:, 2] - data['stru'].lf_3pts[:, 1]
                c_norm_vec = c_vec / (c_vec.norm(dim=-1, keepdim=True) + eps)
                rec_node_attr = rec_node_attr.to("cuda:0")
                n_norm_vec = n_norm_vec.to("cuda:0")
                c_norm_vec = c_norm_vec.to("cuda:0")
                rec_edge_index = rec_edge_index.to("cuda:0")
                rec_edge_attr_ = rec_edge_attr_.to("cuda:0")
                rec_edge_sh = rec_edge_sh.to("cuda:0")
                rec_intra_update = self.rec_conv_layers[l](torch.cat([rec_node_attr, n_norm_vec, c_norm_vec], dim=-1), rec_edge_index, rec_edge_attr_, rec_edge_sh)
                # print("l==1,rec_intra_update: ")
                # print(rec_intra_update.shape)  # (1108,28)

            else:
                rec_node_attr =rec_node_attr.to("cuda:0")
                rec_edge_index = rec_edge_index.to("cuda:0")
                rec_edge_attr_ = rec_edge_attr_.to("cuda:0")
                rec_edge_sh = rec_edge_sh.to("cuda:0")
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
                # print("l>1,rec_intra_update: ")
                # print(rec_intra_update.shape)  # (1108,40),(1108,56),(1108,56),(1108,56)

            rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
            rec_node_attr = rec_node_attr + rec_intra_update

        res_tr_pred = (rec_node_attr[:, self.ns:self.ns+self.nv*3]).view(rec_node_attr.shape[0], -1, 3).mean(1)
        res_rot_pred = (rec_node_attr[:, self.ns+self.nv*3:self.ns+(self.nv+self.nv)*3]).view(rec_node_attr.shape[0], -1, 3).mean(1)
        res_chi_pred = torch.cat([rec_node_attr[:, :self.ns], rec_node_attr[:, -self.ns:]], dim=-1)

        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['res_tr'])

        # fix the magnitude of translational and rotational score vectors
        res_tr_norm = torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1)
        res_tr_pred = res_tr_pred.to("cuda:0")
        res_tr_norm = res_tr_norm.to("cuda:0")
        res_tr_norm = res_tr_norm.to("cuda:0")
        data.graph_sigma_emb = data.graph_sigma_emb.to("cuda:0")
        res_tr_pred = res_tr_pred / (res_tr_norm+eps) * self.res_tr_final_layer(torch.cat([res_tr_norm, data.graph_sigma_emb], dim=1))
        res_rot_norm = torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1)
        res_rot_pred = res_rot_pred / (res_rot_norm+eps) * self.res_rot_final_layer(torch.cat([res_rot_norm, data.graph_sigma_emb], dim=1))

        res_chi_pred = self.res_chi_final_layer(res_chi_pred)

        # print(res_tr_pred.shape)   # (1144,3)
        # print(res_rot_pred.shape)  # (1144,3)
        # print(res_chi_pred.shape)  # (1144,5)
        # print("okokokokokokokokokokokok")
        return res_tr_pred, res_rot_pred, res_chi_pred

    def build_rec_conv_graph(self, data):
        # builds the structure initial node and edge embeddings
        data['stru'].node_sigma_emb = self.timestep_emb_func(data['stru'].node_t['tr'])  # tr rot and tor noise is all the same
        node_attr = torch.cat([data['stru'].x, data['stru'].chis.sin()*data['stru'].chi_masks, data['stru'].chis.cos()*data['stru'].chi_masks, data['stru'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        # pos是c_alpha_coords
        edge_index = radius_graph(data['stru'].pos, self.rec_max_radius, data['stru'].batch, max_num_neighbors=self.c_alpha_max_neighbors)
        edge_index = edge_index[[1, 0]]
        src, dst = edge_index
        edge_vec = data['stru'].pos[dst.long()] - data['stru'].pos[src.long()]
        # 边距离嵌入为高斯分布的特征，通过高斯函数对输入距离进行平滑处理，生成一组新的特征
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['stru'].node_sigma_emb[edge_index[0].long()].to(edge_length_emb.device)

        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh
