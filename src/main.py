from email.mime import base
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import scipy.sparse as sp
import pickle 
import copy

from torch_sparse.tensor import cpu

# from utils import load_graphs, load_feats

from models.wrap_models import AttackModel

from src.pgd_attack import TDPGD
from dataset import Dataset

from utils import * #normalize_adjs, sparse_mx_to_torch_sparse_tensor, get_gpu_info
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, top_k_accuracy_score
import sys
from metrics import *
import pickle as pkl
import time
import os
from copy import deepcopy

def get_attack_model (cmd_args, num_feats, num_nodes, nclasses, saved_model, model_task, train_edges=None, train_labels=None, neg_sampling=True): #, embs=False):
    with open('{}/{}_{}_{}_{}.pkl'.format(saved_model, model_task, cmd_args.num_graphs, cmd_args.context, cmd_args.target_ts), 'rb') as f:
        base_args = pickle.load(f)
    
    base_args['device'] = cmd_args.device
    base_args['task'] = cmd_args.task
    base_args['model_name'] = cmd_args.model_name
    base_args['GPU_ID'] = int(cmd_args.device[5:]) if ('cuda' in cmd_args.device) else -1
    base_args['target_ts'] = cmd_args.target_ts
    base_args['historical_len'] = cmd_args.context
    base_args['nnodes'] = num_nodes
    base_args['nclasses'] = nclasses
    base_args['task'] = cmd_args.task
    base_args['nfeats'] = num_feats
    base_args['neg_sampling'] = neg_sampling
    base_args['train_edges'] = train_edges
    base_args['train_labels'] = train_labels
    # base_args['gcn_normalize'] = False

    return AttackModel(base_args) #if (not (embs)) else AttackEmbModel(base_args)
    # return DynGraphVictim(**base_args).to(device)

def load_base_model(cmd_args, num_feats, num_nodes, nclasses=None, train_edges=None, train_labels=None):
    assert cmd_args.saved_model is not None
    print (cmd_args.saved_model)

    model_task = "modelnc" if cmd_args.task == 'node_classification' else 'model'

    base_model = get_attack_model (cmd_args, num_feats, num_nodes, nclasses, cmd_args.saved_model, model_task, train_edges=train_edges, 
                                    train_labels=train_labels, neg_sampling=cmd_args.neg_sampling)
    if torch.cuda.is_available() and ('cuda' in cmd_args.device):
        map_location=lambda storage, loc: storage.cuda(int(cmd_args.device.split("cuda:")[1]))
    else:
        map_location='cpu'
    try:
        base_model.model.load_state_dict(torch.load("{}/{}_{}_{}_{}.pt".format(
                                        cmd_args.saved_model, model_task, cmd_args.num_graphs, cmd_args.context, cmd_args.target_ts), 
                                        map_location=map_location))
    except:
        if (cmd_args.neg_sampling):
            base_model.model.load_state_dict(torch.load("{}/{}_ns_{}_len{}_ts{}.pt".format(
                                            cmd_args.saved_model, model_task, cmd_args.num_graphs, cmd_args.context,  cmd_args.target_ts), 
                                            map_location=map_location))
        else:
            base_model.model.load_state_dict(torch.load("{}/{}_{}_len{}_ts{}.pt".format(
                                            cmd_args.saved_model, model_task, cmd_args.num_graphs, cmd_args.context, cmd_args.target_ts), 
                                            map_location=map_location))
    base_model.eval()
    return base_model.to(torch.double).to(cmd_args.device)

def main (cmd_args):
    print(cmd_args)
    print(cmd_args, file=sys.stderr)
    # Set random seed
    seed = cmd_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data needs to be updated according to DySAT, let's say.
    # graphs, adjs = load_graphs(cmd_args.dataset)
    # if cmd_args.featureless:
    #     feats = [sp.identity(adjs[num_graphs - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
    #             x.shape[0] <= adjs[num_graphs - 1].shape[0]]
    # else:
    #     feats = load_feats(cmd_args.dataset)

    num_graphs = cmd_args.num_graphs
    task = cmd_args.task
    device = cmd_args.device
    target_ts = cmd_args.target_ts
    context = cmd_args.context
    ntargets = cmd_args.ntargets

    data = Dataset(root='data', name=cmd_args.dataset, context=context, task=task, num_graphs=num_graphs, dyn_feats=cmd_args.dyn_feats, seed=seed,
                   ntargets=ntargets, featureless=cmd_args.featureless, directed=not(cmd_args.undirected), device='cpu' if (cmd_args.large_graph) else device)   
    target_snapshot = cmd_args.target_ts #num_graphs - 1 # if (task == "link_prediction") else num_graphs-1
    if (task == "node_classification"):
        data.data_split(target_snapshot, train_p=0.6, val_p=0.2, test_p=0.2, sampling=cmd_args.sampling, num_samples=cmd_args.num_samples)
    elif (task == "edge_classification"):
        data.data_split(target_snapshot, train_p=0.1, val_p=0.1, test_p=0.8)
    elif (task == "link_prediction"):
        # data.link_split(target_snapshot, val_p=0.3, test_p=0.4, sampling=cmd_args.sampling, num_samples=cmd_args.num_samples)
        data.link_split(target_snapshot, val_p=0.0, test_p=1.0, sampling=cmd_args.sampling, num_samples=cmd_args.num_samples)

    # 
    # features are also list of timestamped features, dynamic --
    if not cmd_args.dyn_feats:
        data.normalize_feats()

    data.features = torch.tensor(data.features).to(device)
    # data.features = sparse_mx_to_torch_sparse_tensor(data.features) #.to(device)

    # # # # TO FIX -- ONLY Because dysat is using such an identity feature map. But infeasible/useless in general. 
    if cmd_args.featureless and cmd_args.model_name == 'DySAT_pytorch': 
        data.features = torch.tensor(np.array(sp.identity(data.adjs[cmd_args.num_graphs - 1].shape[0]).tocsr().todense(), dtype=np.float32)).to(cmd_args.device)

    print (data.features.shape)
    # data.adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in data.adjs]
    # data.to_sparseTensor(num_ts=num_graphs)
    data.to_tg_data(num_ts=num_graphs, island=True)
    context_graphs = data.graphs[target_ts-context:target_ts]
    print(int(data.graphs[target_ts].edge_index.max()) + 1)
    print([(int(g.edge_index.max()) + 1) for g in context_graphs])
    # 
    # evasive attack
    idx_targets, y_targets = data.test_mask, data.test_y
    classes = np.unique(y_targets)
    nclasses = len(classes)
    print (len(idx_targets), data.graphs[0].x.shape[1], len(classes))
    # idx_targets = np.concatenate((np.array(pos_links), idx_targets[y_targets==0][:len(pos_links)]))
    # y_targets = np.concatenate((np.ones(len(pos_links)), np.zeros(len(pos_links))))

    # Load model
    victim_model = load_base_model(cmd_args, num_feats=data.graphs[0].x.shape[1], num_nodes=data.max_nodes, 
                                    nclasses=nclasses, train_edges=data.train_mask, train_labels=data.train_y)
    print (victim_model.device)
    orig_embs = (victim_model(context_graphs, idx_targets=idx_targets) if (cmd_args.large_graph) else victim_model(context_graphs)).detach()
    print (orig_embs.shape)
    if (ntargets > 1):
        orig_probs = victim_model.predict(
            orig_embs, 
            idx_targets.reshape(idx_targets.shape[0]*idx_targets.shape[1], idx_targets.shape[2])
        ).detach().cpu().numpy().reshape(idx_targets.shape[:2])
    else:
        orig_probs = victim_model.predict(orig_embs, idx_targets).detach().cpu().numpy()
    orig_embs = orig_embs.to('cpu') if (cmd_args.large_graph) else orig_embs
    # 
    # With accuracy:::
    # 
    # pred_y = torch.sigmoid(victim_model.predict(orig_embs, idx_targets)).detach().cpu().numpy()
    # threshold = 0.925 # 2.51230562398
    # orig_prediction = (np.where(pred_y < threshold, 0, 1) == y_targets)
    # print ("Original accuracy:", orig_prediction.mean())
    # wrong_preds, total_targets = (~orig_prediction).sum(), len(orig_prediction)
    # print (wrong_preds, total_targets)
    # idx_targets = idx_targets[orig_prediction]
    # y_targets = y_targets[orig_prediction]
    # av_perb_acc = 0.0
    # 
    # With rocauc:::
    print (y_targets.shape, orig_probs.shape)
    total_edges = y_targets.shape[0]*ntargets
    if task == 'link_prediction':
        if (ntargets > 1):
            orig_metric = av_roc_targets(y_targets, orig_probs)
            # orig_metric = roc_auc_score(y_targets.reshape(total_edges), orig_probs.reshape(total_edges))
        else:
            orig_metric = roc_auc_score(y_targets, orig_probs) 
    elif task == 'node_classification':
        if (nclasses == 2):
            orig_metric = roc_auc_score (y_targets, orig_probs)
        else:
            print (confusion_matrix(y_targets, orig_probs.argmax(axis=-1)))
            orig_metric = accuracy_score (y_targets, orig_probs.argmax(axis=-1))
    
    print (orig_probs)
    print ("Orig AUCROC: {}".format(orig_metric))

    pred_attkd = []
    total_time = 0.0
    for target_id in range(idx_targets.shape[0]):
        target_nds = idx_targets[target_id]
        target_label = y_targets[target_id]
        if (cmd_args.budget is not None):
            budget = cmd_args.budget
        elif (cmd_args.budget_perc is not None) and (not (cmd_args.bp_tgspec)):
            tgdeg_sum = num_conn_nodes (context_graphs, target_nds, data.max_nodes)
            budget = int (tgdeg_sum * cmd_args.budget_perc)
            # tgdeg_sum = deg_sum (context_graphs, target_nds, data.max_nodes)
            # budget_s = int(tgdeg_sum * cmd_args.budget_perc)
        # print ('Target_id: {}, Budget: {}'.format(target_id, budget))
        # print ('Target_id: {}, Budget: {}'.format(target_id, budget_s))

        if (ntargets > 1) and (cmd_args.seq_tg_attk):
            mod_graphs = deepcopy(context_graphs)
            if ("deg" in cmd_args.seq_order):
                # print (target_nds, target_label)
                deg_desc = True if (cmd_args.seq_order[4:] == "desc") else False
                map_ids, target_nds, target_label = deg_sorted_links(mod_graphs, target_nds, target_label, data.max_nodes, descending=deg_desc)
                # print (target_nds, target_label)
            elif ("prob" in cmd_args.seq_order):
                desc = True if (cmd_args.seq_order[5:] == "desc") else False
                map_ids, target_nds, target_label = prob_sorted_links(mod_graphs, victim_model, target_nds, target_label, large_graph=cmd_args.large_graph, descending=desc)
            elif ("ovlp" in cmd_args.seq_order):
                desc = True if (cmd_args.seq_order[5:] == "desc") else False
                map_ids, target_nds, target_label = ovlp_sorted_links(mod_graphs, target_nds, target_label, data.max_nodes, descending=desc)
            elif ("rand" in cmd_args.seq_order):
                link_ids = np.arange(len(target_nds))
                np.random.shuffle(link_ids)
                map_ids, target_nds, target_label = target_nds[link_ids], target_label[link_ids]
            else:
                map_ids, target_nds, target_label = np.arange(target_nds.shape[0]), target_nds, target_label
            inv_ids = inv_map(map_ids)
            perbs = []
            for ntg in range(ntargets):
                if (cmd_args.method == "pgd"):
                    attacker = TDPGD(model=victim_model, num_graphs=context, ntg_nodes=2, khop=cmd_args.khop, thresh_wt=cmd_args.thresh_wt, args_device=device, inits=cmd_args.inits,
                                nclasses=nclasses, large_graph=cmd_args.large_graph, directed=not(cmd_args.undirected), nnodes=data.max_nodes, loss_type=cmd_args.loss_type, device=device)
                target_time = time.time()
                nedges = []
                for graph in mod_graphs:
                    ei_indices = torch.zeros_like(graph.edge_index[0], dtype=bool)
                    for n in target_nds[ntg]:
                        ei_indices = torch.logical_or(graph.edge_index[0] == n, ei_indices)
                    nedges.append(ei_indices.sum().item())
                print (nedges)
                # embs = victim_model(mod_graphs, idx_targets=target_nds[ntg], to_predict=True).to('cpu')
                # try:
                if (cmd_args.constraint == "budget"):
                    if (cmd_args.bp_tgspec):
                        tgdeg_sum = num_conn_nodes (context_graphs, target_nds[ntg], data.max_nodes)
                        tg_budget = int (tgdeg_sum * cmd_args.budget_perc)
                    else:
                        tg_budget = budget // ntargets
                    print (target_label[ntg], target_nds[ntg], tg_budget, nedges)
                    attacker.attack(mod_graphs, target_label[ntg], target_nds[ntg], constraint=cmd_args.constraint,  constr_vars=[tg_budget], 
                                    orig_embs=orig_embs, lambda1=cmd_args.lambda1, epochs=cmd_args.num_steps, lr_init=cmd_args.lr_init, use_optim=cmd_args.use_optim)
                elif (cmd_args.constraint == "noise"):
                    attacker.attack(mod_graphs, target_label[ntg], target_nds[ntg], constraint=cmd_args.constraint,  constr_vars=[cmd_args.epsilon, cmd_args.epsilon1], 
                                    orig_embs=orig_embs, lambda1=cmd_args.lambda1, epochs=cmd_args.num_steps, lr_init=cmd_args.lr_init, use_optim=cmd_args.use_optim)
                     
                total_time += time.time() - target_time
                mod_graphs = [graph.to('cpu') for graph in attacker.modified_graphs] if (cmd_args.large_graph) else attacker.modified_graphs
                # except:
                #     pass 
                # del attacker
                perbs.append(torch.where(attacker.adj_changes if (cmd_args.method == "pgd") else attacker.perturbed_edge_weight))
            print (total_time)
        else:
            if (cmd_args.method == "pgd"):
                attacker = TDPGD(model=victim_model, num_graphs=context, ntg_nodes=ntargets*2, khop=cmd_args.khop, thresh_wt=cmd_args.thresh_wt, large_graph=cmd_args.large_graph,
                            nclasses=nclasses, inits=cmd_args.inits, directed=not(cmd_args.undirected), nnodes=data.max_nodes, loss_type=cmd_args.loss_type, device=device, args_device=device)
            target_time = time.time()
            if (cmd_args.constraint == "budget"):
                attacker.attack(context_graphs, target_label, target_nds, constraint=cmd_args.constraint,  constr_vars=[cmd_args.budget], 
                                orig_embs=orig_embs, lambda1=cmd_args.lambda1, epochs=cmd_args.num_steps, lr_init=cmd_args.lr_init, use_optim=cmd_args.use_optim)
            elif (cmd_args.constraint == "noise"):
                attacker.attack(context_graphs, target_label, target_nds, constraint=cmd_args.constraint, constr_vars=[cmd_args.epsilon, cmd_args.epsilon1], 
                                orig_embs=orig_embs, lambda1=cmd_args.lambda1, epochs=cmd_args.num_steps, lr_init=cmd_args.lr_init, use_optim=cmd_args.use_optim)
            total_time += time.time() - target_time
            print (total_time)
            mod_graphs = attacker.modified_graphs

        mod_graphs = [graph.to('cpu') for graph in attacker.modified_graphs] if (cmd_args.large_graph) else attacker.modified_graphs
        embs = victim_model(mod_graphs, idx_targets=idx_targets) if (cmd_args.large_graph) else victim_model(mod_graphs)
        pred = victim_model.predict(embs, target_nds)
        try:
            pred_attkd.append(pred.item())
        except:
            pred_attkd.append(pred.detach().cpu().numpy())
        
        if (ntargets > 1) and (cmd_args.seq_tg_attk):
            pred = pred[inv_ids]
            perturbations = [None] * len(perbs[0])
            for j in range(len(perbs[0])):
                perturbations[j] = torch.cat([perb[j] for perb in perbs])
            perturbations = tuple (perturbations)
        else:
            perturbations = torch.where(attacker.adj_changes) if (cmd_args.method == "pgd") else torch.where(attacker.perturbed_edge_weight)

        
        with torch.no_grad():
            # "Pred perb_acc, 
            embs = embs.to('cpu') if (cmd_args.large_graph) else embs
            print (embs.device, mod_graphs[0].edge_index.device)
            if (perturbations[0].shape[0] < 100):
                print ("Target_id: {}, Perturbations: {}".format(target_id, perturbations))
            else:
                print ("Target_id: {}, Perturbation size: {}".format(target_id, perturbations[0].shape[0]))
            if task == 'link_prediction':
                print ("dz': {}, dz'/dz: {}, K: {}, E: {}, dz'-dz: {}".format(
                            attacker.smooth_loss (embs, mod_graphs, normalized_return=False).item(), 
                            attacker.smooth_loss (embs, mod_graphs, normalized_return=True),
                            get_stability_K (target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop), 
                            get_stability_E (target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop),
                            get_dzDiff_range(target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs, 
                                            context_graphs, k=cmd_args.khop)))
            elif task == 'node_classification':
                print ("dz': {}, dz'/dz: {}, K: {}, E: {}, dz'-dz: {}".format(
                            attacker.smooth_loss (embs, mod_graphs, normalized_return=False).item(), 
                            attacker.smooth_loss (embs, mod_graphs, normalized_return=True),
                            get_stability_K (np.array([target_nds]), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop), 
                            get_stability_E (np.array([target_nds]), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop),
                            get_dzDiff_range(np.array([target_nds]), embs, mod_graphs, orig_embs, 
                                            context_graphs, k=cmd_args.khop)))

    #     av_perb_acc += perb_acc

    # av_perb_acc /= total_targets
    # print (av_perb_acc)

    print ("Original Probs:", orig_probs)
    print ("Attacked Probs:", np.stack(pred_attkd))
    if task == 'link_prediction':
        if (ntargets > 1):
            print ("AUCROC after perturbation: {}".format(av_roc_targets(y_targets, np.stack(pred_attkd))))
        else:
            print ("AUCROC after perturbation: {}".format(roc_auc_score(y_targets.reshape(total_edges), pred_attkd)))
    elif task == 'node_classification':
        if (nclasses == 2):
            print ("AUCROC after perturbation: {}".format(roc_auc_score (y_targets, pred_attkd)))
        else:
            print (confusion_matrix(y_targets, np.stack(pred_attkd).argmax(axis=-1)))
            print ("Accuracy after perturbation: {}".format(accuracy_score (y_targets, np.stack(pred_attkd).argmax(axis=-1))))
        # get_roc_score_t(idx_targets[y_targets == 1], idx_targets[y_targets == 0], source_emb, target_emb)))
    print ("Total time taken:", total_time)

# Orig AUCROC: 0.9826
# AUCROC after perturbation: 0.82495



# # Testing after attack
# # adjs_norm = normalize_adjs(modified_adjs)
# norm_diff = 0.0
# from torch_sparse import coalesce
# from torch_geometric.utils import to_scipy_sparse_matrix
# import scipy
    
# with torch.no_grad():
#     t = 0
#     for graph, mod_graph in zip (context_graphs, attacker.modified_graphs):
#         # basically find the diff edge_index and edge_weight
#         m, n = torch.max(torch.max(graph.edge_index, dim=1).values, torch.max(mod_graph.edge_index, dim=1).values) + 1
#         dA_ei = torch.cat((mod_graph.edge_index, graph.edge_index), dim=1)
#         dA_ew = torch.cat((mod_graph.edge_weight, -graph.edge_weight))
#         dA_ei, dA_ew = coalesce (dA_ei, dA_ew, m=m, n=n)
#         # mod_adj = scipy.sparse.csr_matrix(to_scipy_sparse_matrix(mod_graph.edge_index, mod_graph.edge_weight, num_nodes=143))
#         # orig_adj = scipy.sparse.csr_matrix(to_scipy_sparse_matrix(graph.edge_index, graph.edge_weight, num_nodes=143))
#         # print (torch.nonzero(attacker.adj_changes[t, 0, :, 0]), orig_adj[target_nds[0]].nonzero()[1], mod_adj[target_nds[0]].nonzero()[1], return_complem (graph.edge_index, target_nds[0], 0)[mod_adj[target_nds[0]].nonzero()[1]])
#         # print (torch.nonzero(attacker.adj_changes[t, 0, :, 1]), orig_adj[:,target_nds[0]].nonzero()[0], mod_adj[:,target_nds[0]].nonzero()[0], return_complem (graph.edge_index, target_nds[0], 1)[mod_adj[:,target_nds[0]].nonzero()[0]])
#         # print (torch.nonzero(attacker.adj_changes[t, 1, :, 0]), orig_adj[target_nds[1]].nonzero()[1], mod_adj[target_nds[1]].nonzero()[1], return_complem (graph.edge_index, target_nds[1], 0)[mod_adj[target_nds[1]].nonzero()[1]])
#         # print (torch.nonzero(attacker.adj_changes[t, 1, :, 1]), orig_adj[:,target_nds[1]].nonzero()[0], mod_adj[:,target_nds[1]].nonzero()[0], return_complem (graph.edge_index, target_nds[1], 1)[mod_adj[:,target_nds[1]].nonzero()[0]])
#         # print (t, attacker.adj_changes[t].sum().item(), torch.abs(dA_ew).sum().detach().item())
#         if (cmd_args.undirected):
#             norm_diff += torch.abs(dA_ew).sum().detach().item()/2
#         else:
#             norm_diff += torch.abs(dA_ew).sum().detach().item()
#         t += 1
# print (attacker.adj_changes.sum().item(), norm_diff)
# # Why are these different ??