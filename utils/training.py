# coding=utf-8
"""
@author:cxy
@file: training.py
@date: 2024/4/4 10:13
"""
import copy

import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from datasets.process import get_align_rotran
from utils.diffusion_utils import set_time, get_t_schedule
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader, DataListLoader
from Bio.PDB import PDBParser, MMCIFParser
from utils.sampling import randomize_position, sampling
from utils.utils import ListDataset


def compute_RMSD(a, b):
    # correct rmsd calculation.
    return np.sqrt((((a-b)**2).sum(axis=-1)).mean())


def loss_function(res_tr_pred, res_rot_pred, res_chi_pred, data, t_to_sigma, device, res_tr_weight=1, res_rot_weight=1, res_chi_weight=1, apply_mean=True, no_torsion=False, train_score=False, finetune=False):
    mean_dims = (0, 1) if apply_mean else 1
    data_t = [data.complex_t[noise_type] for noise_type in ['res_tr', 'res_rot', 'res_chi']]
    res_tr_sigma, res_rot_sigma, res_chi_sigma = t_to_sigma(*data_t)
    res_loss_weight = data.res_loss_weight

    # local translation component
    res_tr_score = data.res_tr_score
    res_tr_loss = torch.nn.L1Loss(reduction='none')(res_tr_pred.cpu(), res_tr_score).mean(dim=1) * res_loss_weight.squeeze(1) * 3.0
    res_tr_base_loss = (res_tr_score).abs().mean(dim=1).detach() * res_loss_weight.squeeze(1) * 3.0
    if apply_mean:
        res_tr_loss = res_tr_loss.mean()
        res_tr_base_loss = res_tr_base_loss.mean()

    # local rotation component
    res_rot_score = data.res_rot_score
    res_rot_loss_pos = (torch.nn.L1Loss(reduction='none')(res_rot_pred.cpu(), res_rot_score)).mean(dim=1)
    res_rot_loss = res_rot_loss_pos * res_loss_weight.squeeze(1) * 15.0
    res_rot_base_loss = (res_rot_score.abs()).mean(dim=1).detach() * res_loss_weight.squeeze(1) * 15.0  # 为什么要乘以15？
    if apply_mean:
        res_rot_loss = res_rot_loss.mean()
        res_rot_base_loss = res_rot_base_loss.mean()

    res_chi_score = data.res_chi_score
    res_chi_mask = data['stru'].chi_masks
    res_chi_mask = res_chi_mask[:, [0, 2, 4, 5, 6]]
    res_chi_symmetry_mask = data['stru'].chi_symmetry_masks
    res_chi_symmetry_mask = res_chi_symmetry_mask.bool()
    res_chi_loss = 1-(res_chi_pred.cpu()-res_chi_score).cos()
    res_chi_symmetry_loss = 1-(res_chi_pred.cpu()-res_chi_score-np.pi).cos()
    res_chi_loss[res_chi_symmetry_mask] = torch.minimum(res_chi_loss[res_chi_symmetry_mask],res_chi_symmetry_loss[res_chi_symmetry_mask])
    res_chi_loss = (res_chi_loss*res_loss_weight*res_chi_mask).sum(dim=mean_dims) / (res_chi_mask.sum(dim=mean_dims)+1e-12) * 3.0
    res_chi_base_loss = ((1-(res_chi_score).cos())*res_loss_weight*res_chi_mask).sum(dim=mean_dims) / (res_chi_mask.sum(dim=mean_dims).detach()+1e-12) * 3.0

    if not apply_mean:
        rec_batch = data['stru'].batch
        res_tr_loss = scatter_mean(res_tr_loss, rec_batch)
        res_rot_loss = scatter_mean(res_rot_loss, rec_batch)
        res_chi_loss = scatter_mean(res_chi_loss, rec_batch)
        res_tr_base_loss = scatter_mean(res_tr_base_loss, rec_batch)
        res_rot_base_loss = scatter_mean(res_rot_base_loss, rec_batch)
        res_chi_base_loss = scatter_mean(res_chi_base_loss, rec_batch)

    loss = res_tr_loss * res_tr_weight + res_rot_loss * res_rot_weight + res_chi_loss * res_chi_weight
    base_loss = res_tr_base_loss * res_tr_weight + res_rot_base_loss * res_rot_weight + res_chi_base_loss * res_chi_weight

    return loss, res_tr_loss.detach(), res_rot_loss.detach(), res_chi_loss.detach(), base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, train_score=False, finetune=False):
    model.train()
    meter = AverageMeter(['loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_loss', 'base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'])
    bar = tqdm(loader, total=len(loader)) 
    train_loss = 0.0
    train_num = 0.0

    for data in bar:
        optimizer.zero_grad()
        try:
        # print(data["name"])
            res_tr_pred, res_rot_pred, res_chi_pred = model(data)
            loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss = \
                loss_fn(res_tr_pred, res_rot_pred, res_chi_pred, data=data, t_to_sigma=t_to_sigma, device=device, train_score=train_score, finetune=finetune)
            loss.backward()
            optimizer.step()
            ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss])
            train_loss += loss.item()
            train_num += 1
            bar.set_description('loss: %.4f' % (train_loss/train_num))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                # raise e
                continue
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_loss', 'base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_losss', 'base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                res_tr_pred, res_rot_pred, res_chi_pred = model(data)

            loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss = \
                loss_fn(res_tr_pred, res_rot_pred, res_chi_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss])

            if test_sigma_intervals > 0:
                structure_t_res_tr, structure_t_res_rot, structure_t_res_chi = [torch.cat([d.complex_t[noise_type] for d in data]) for
                                                              noise_type in ['res_tr', 'res_rot', 'res_chi']]
                sigma_index_res_tr = torch.round(structure_t_res_tr.cpu() * (10 - 1)).long()
                sigma_index_res_rot = torch.round(structure_t_res_rot.cpu() * (10 - 1)).long()
                sigma_index_res_chi = torch.round(structure_t_res_chi.cpu() * (10 - 1)).long()
                meter_all.add([loss.cpu().detach(), res_tr_loss, res_rot_loss, res_chi_loss, base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found - skipping batch')
                continue
            else:
                raise e
    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch(model, structure_graphs, device, t_to_sigma, args):
    model.eval()
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    res_tr_schedule, res_rot_schedule, res_chi_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(structure_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []
    si = 0
    # orig_structure_graphs = []
    for orig_structure_graph in tqdm(loader):
        # orig_structure_graphs.append(orig_structure_graph)
        data_list = [copy.deepcopy(orig_structure_graph)]
        si += 1
        predictions_list = []
        if (si+1) % args.sample_batch_size != 0 and si != len(structure_graphs):
            continue
        elif len(data_list) > 0:
            randomize_position(data_list, args.no_torsion, False, args.res_tr_sigma_max, args.res_rot_sigma_max)
            predictions_one = None
            data_list_one = None
            data_list_step = []
            failed_convergence_counter = 0
            # while predictions_list == None:
            # model = model.module if device.type == 'cuda' else model,
            try:
                predictions_one, data_list_one = sampling(data_list=data_list, model=model,
                                                         inference_steps=args.inference_steps, ode=True,
                                                         res_tr_schedule=res_tr_schedule, res_rot_schedule=res_rot_schedule, res_chi_schedule=res_chi_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
                predictions_list.append(predictions_one)
                data_list_step.append(data_list_one)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                elif 'no cross edge found' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: no cross edge found - skipping the complex')
                        break
                    print('| WARNING: no cross edge found - trying again with a new sample')
                else:
                    raise e
        for i, data in enumerate(predictions_list):
            pred_rec_pos_ = []
            orig_structure_graph_ = np.expand_dims(orig_structure_graph['stru'].pos.cpu().numpy(), axis=0)
            pred_rec_pos = data[0]['stru'].pos.cpu().numpy()  # numpy.ndarray
            pred_rec_pos_.append(pred_rec_pos)
            # print(pred_rec_pos.shape)
            # print(orig_structure_graph['stru'].pos.cpu().numpy().shape)
            # print("111111")
            squared_diff = np.sum((pred_rec_pos_ - orig_structure_graph_) ** 2, axis=2)
            mean_squared_diff = np.mean(squared_diff, axis=1)
            rmsd = np.sqrt(mean_squared_diff)
            rmsds.append(rmsd)
        data_list = []
        orig_structure_graphs = []
    rmsds = np.array(rmsds)
    losses = {
        'RMSD': (rmsds.sum()/len(rmsds))
    }
    return losses


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                # print(type_idx)
                # print(v)
                # print("1234")
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def finetune_epoch(model, loader, device, t_to_sigma, args, optimizer, loss_fn, ema_weights):
    model.train()
    meter = AverageMeter(['loss'])

    bar = tqdm(loader, total=len(loader))
    train_loss = 0.0
    train_num = 0.0
    for data in bar:
        # if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
        #     print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
            for d in data:
                set_time(d, t_res_tr, t_res_rot, t_res_chi, 1, False, None)
            # TODO 注意这里None的数量需要确定一下！！！！
            loss = loss_fn(None, None, None, None, None, None, None, data=data, t_to_sigma=t_to_sigma, device=device, finetune=True)
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach()])
            train_loss += loss.item()
            train_num += 1
            bar.set_description('loss: %.4f' % (train_loss/train_num))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                # raise e
                continue
    return meter.summary()

