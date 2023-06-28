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

from wrap_models import AttackModel

from args import cmd_args
# from deeprobust.graph import utils 
from utils import * #normalize_adjs, sparse_mx_to_torch_sparse_tensor, get_gpu_info
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, top_k_accuracy_score
import sys
from metrics import *
from evals import *
import pickle as pkl
import time
import os
from copy import deepcopy

from baselines.random.rand_attack import RandAttack
from baselines.degree.deg_attack import DegAttack
from dataset import Dataset
from tqdm import tqdm

def get_attack_model (num_feats, num_nodes, nclasses, saved_model, model_task, train_edges=None, train_labels=None, neg_sampling=True): #, embs=False):
    with open('{}/{}_{}_{}_{}.pkl'.format(saved_model, model_task, cmd_args.num_graphs, cmd_args.context, cmd_args.target_ts), 'rb') as f:
        base_args = pickle.load(f)
    
    base_args['device'] = cmd_args.device
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

def load_base_model(num_feats, num_nodes, nclasses=None, train_edges=None, train_labels=None):
    assert cmd_args.saved_model is not None
    print (cmd_args.saved_model)

    model_task = "modelnc" if cmd_args.task == 'node_classification' else 'model'

    base_model = get_attack_model (num_feats, num_nodes, nclasses, cmd_args.saved_model, model_task, train_edges=train_edges, 
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

if __name__ == '__main__':
    print(cmd_args)
    print(cmd_args, file=sys.stderr)
    # Set random seed
    seed = cmd_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_graphs = cmd_args.num_graphs
    task = cmd_args.task
    device = cmd_args.device
    target_ts = cmd_args.target_ts
    context = cmd_args.context
    ntargets = cmd_args.ntargets

    # would need to change the definitions of "Dataset" and "preprocess" inside the DeepRobust library
    data = Dataset(root='data', name=cmd_args.dataset, context=context, task=task, num_graphs=num_graphs, dyn_feats=cmd_args.dyn_feats,
                   ntargets=ntargets, featureless=cmd_args.featureless, directed=not(cmd_args.undirected), device='cpu' if (cmd_args.large_graph) else device)   
    target_snapshot = cmd_args.target_ts #num_graphs - 1 # if (task == "link_prediction") else num_graphs-1
    if (task == "node_classification"):
        data.data_split(target_snapshot, train_p=0.6, val_p=0.2, test_p=0.2, sampling=cmd_args.sampling, num_samples=cmd_args.num_samples)
    elif (task == "edge_classification"):
        data.data_split(target_snapshot, train_p=0.1, val_p=0.1, test_p=0.8)
    elif (task == "link_prediction"):
        data.link_split(target_snapshot, val_p=0.3, test_p=0.4, sampling=cmd_args.sampling, num_samples=cmd_args.num_samples)
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
    nfeats = data.features.shape[-1]
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
    if not cmd_args.save_only:
        victim_model = load_base_model(num_feats=data.graphs[0].x.shape[1], num_nodes=data.max_nodes, 
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
    if cmd_args.save_only:
        saved_perbs = {}
    elif cmd_args.analyze_only:
        saved_perbs = pkl.load(open(cmd_args.save_perbs_file, "rb"))
    for target_id in tqdm(range(idx_targets.shape[0])):
        target_nds = idx_targets[target_id]
        target_label = y_targets[target_id]
        if (cmd_args.budget is not None):
            budget = cmd_args.budget
        elif (cmd_args.budget_perc is not None) and (not (cmd_args.bp_tgspec)):
            tgdeg_sum = num_conn_nodes (context_graphs, target_nds, data.max_nodes)
            budget = int (tgdeg_sum * cmd_args.budget_perc)

        if (not cmd_args.analyze_only) and (ntargets > 1):
            mod_graphs = deepcopy(context_graphs)
            if ("deg" in cmd_args.seq_order):
                deg_desc = True if (cmd_args.seq_order[4:] == "desc") else False
                map_ids, target_nds, target_label = deg_sorted_links(mod_graphs, target_nds, target_label, data.max_nodes, descending=deg_desc)
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
                if cmd_args.method == "random":
                    attacker = RandAttack(num_graphs=context, ntg_nodes=2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                                nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
                elif cmd_args.method == "degree":
                    attacker = DegAttack(num_graphs=context, ntg_nodes=2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                                nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
                target_time = time.time()
                nedges = []
                for graph in mod_graphs:
                    ei_indices = torch.zeros_like(graph.edge_index[0], dtype=bool)
                    for n in target_nds[ntg]:
                        ei_indices = torch.logical_or(graph.edge_index[0] == n, ei_indices)
                    nedges.append(ei_indices.sum().item())
                print (nedges)
                if (cmd_args.constraint == "budget"):
                    if (cmd_args.bp_tgspec):
                        tgdeg_sum = num_conn_nodes (context_graphs, target_nds[ntg], data.max_nodes)
                        tg_budget = int (tgdeg_sum * cmd_args.budget_perc)
                    else:
                        tg_budget = budget // ntargets
                    print (target_label[ntg], target_nds[ntg], tg_budget, nedges)
                    attacker.attack_budget(mod_graphs, target_nds[ntg], cmd_args.budget)
                elif (cmd_args.constraint == "noise"):
                    attacker.attack_noise(mod_graphs, target_nds[ntg], cmd_args.epsilon, cmd_args.epsilon1)
                total_time += time.time() - target_time
                mod_graphs = [graph.to('cpu') for graph in attacker.modified_graphs] if (cmd_args.large_graph) else attacker.modified_graphs
                # except:
                #     pass 
                # del attacker
                perbs.append(torch.where(attacker.adj_changes['struc']))
            print (total_time)
        elif (not (cmd_args.analyze_only)):
            if cmd_args.method == "random":
                attacker = RandAttack(num_graphs=context, ntg_nodes=ntargets*2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                                nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
            elif cmd_args.method == "degree":
                attacker = DegAttack(num_graphs=context, ntg_nodes=ntargets*2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                            nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
            target_time = time.time()
            if (cmd_args.constraint == "budget"):
                attacker.attack_budget(context_graphs, target_nds, cmd_args.budget)
            elif (cmd_args.constraint == "noise"):
                if (cmd_args.debug):
                    print ("attacking", get_gpu_info(cmd_args.device))
                attacker.attack_noise(context_graphs, target_nds, cmd_args.epsilon, cmd_args.epsilon1)
                if (cmd_args.debug):
                    print ("attacked", get_gpu_info(cmd_args.device))
            total_time += time.time() - target_time
            print (time.time() - target_time)
            mod_graphs = attacker.modified_graphs

        if cmd_args.save_only:
            perturbations = np.where(attacker.adj_changes['struc'].detach().cpu().numpy())
            saved_perbs[target_id] = perturbations
            continue

        if (cmd_args.analyze_only):
            if cmd_args.method == "random":
                attacker = RandAttack(num_graphs=context, ntg_nodes=ntargets*2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                                nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
            elif cmd_args.method == "degree":
                attacker = DegAttack(num_graphs=context, ntg_nodes=ntargets*2, directed=not(cmd_args.undirected), nnodes=data.max_nodes, nclasses=nclasses,
                            nfeats=nfeats, device=device, args_device=device, large_graph=cmd_args.large_graph)
            if (cmd_args.constraint == "budget"):
                attacker.attack_budget(context_graphs, target_nds, cmd_args.budget, analyze_only=True)
            elif (cmd_args.constraint == "noise"):
                attacker.attack_noise(context_graphs, target_nds, cmd_args.epsilon, cmd_args.epsilon1, analyze_only=True)
            attacker.adj_changes['struc'][saved_perbs[target_id]] = 1
            attacker.modified_graphs = attacker.get_modified_graphs(context_graphs)

        mod_graphs = [graph.to('cpu') for graph in attacker.modified_graphs] if (cmd_args.large_graph) else attacker.modified_graphs
        embs = victim_model(mod_graphs, idx_targets=idx_targets) if (cmd_args.large_graph) else victim_model(mod_graphs)
        pred = victim_model.predict(embs, target_nds)
        try:
            pred_attkd.append(pred.item())
        except:
            pred_attkd.append(pred.detach().cpu().numpy())

        if (cmd_args.debug):
            print ("here",  get_gpu_info(cmd_args.device))
        
        if (ntargets > 1) and (cmd_args.seq_tg_attk):
            pred = pred[inv_ids]
            perturbations = [None] * len(perbs[0])
            for j in range(len(perbs[0])):
                perturbations[j] = torch.cat([perb[j] for perb in perbs])
            perturbations = tuple (perturbations)
        else:
            perturbations = torch.where(attacker.adj_changes['struc'])

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
                            get_dz_norm (target_nds.reshape(2*ntargets), embs, mod_graphs, k=cmd_args.khop),
                            get_dznorm_ratio (target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs, 
                                                context_graphs, k=cmd_args.khop),
                            get_stability_K (target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop), 
                            get_stability_E (target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop),
                            get_dzDiff_range(target_nds.reshape(2*ntargets), embs, mod_graphs, orig_embs, 
                                            context_graphs, k=cmd_args.khop)))
            elif task == 'node_classification':
                print (target_nds)
                print ("dz': {}, dz'/dz: {}, K: {}, E: {}, dz'-dz: {}".format(
                            get_dz_norm (np.array([target_nds]), embs, mod_graphs, k=cmd_args.khop), 
                            get_dznorm_ratio (np.array([target_nds]), embs, mod_graphs, orig_embs, 
                                                context_graphs, k=cmd_args.khop),
                            get_stability_K (np.array([target_nds]), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop), 
                            get_stability_E (np.array([target_nds]), embs, mod_graphs, orig_embs=orig_embs, 
                                            orig_graphs=context_graphs, k=cmd_args.khop),
                            get_dzDiff_range(np.array([target_nds]), embs, mod_graphs, orig_embs, 
                                            context_graphs, k=cmd_args.khop)))
        if (cmd_args.debug):
            print ("analyzed, done")
        
        sys.stdout.flush()

    if cmd_args.save_only:
        pkl.dump(saved_perbs, open(cmd_args.save_perbs_file, 'wb'))
        exit()

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
