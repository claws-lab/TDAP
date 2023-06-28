import torch
from torch_sparse import coalesce
import numpy as np

def av_roc_targets (actual, pred):
    from sklearn.metrics import roc_auc_score
    roc_scores = 0.0
    for i in range(actual.shape[0]):
        roc_scores += roc_auc_score(actual[i], pred[i])
    return roc_scores/actual.shape[0]

def norm_diff_graphs (target_nds, graphs, k=2, feats=False):
    def rec_or_targets (ei, target_nds):
        if (target_nds.shape[0] == 1):
            return (ei == target_nds[0])
        else:
            return ((ei == target_nds[0]) | (rec_or_targets(ei, target_nds[1:])))
    if feats:
        dA = torch.zeros(len(graphs) - 1)
        for t in range(1, len(graphs)):
            # basically find the diff edge_index and edge_weight
            dA[t-1] = torch.abs(graphs[t].x[target_nds] - graphs[t-1].x[target_nds]).sum()
        return dA 
    # frobenius by default
    dA = torch.zeros(len(graphs) - 1, device=graphs[0].edge_index.device)
    # frobenius wrt time as well
    for t in range(len(graphs)-1):
        # basically find the diff edge_index and edge_weight
        nbrs = torch.unique(khop_nbrs (graphs[t].edge_index, 
                                      torch.tensor([node for node in target_nds if node in graphs[t].edge_index], 
                                                    dtype=torch.int64, device=graphs[t].edge_index.device), k=k))
        try:
            perm_t, perm_t1 = rec_or_targets(graphs[t].edge_index, nbrs), rec_or_targets(graphs[t+1].edge_index, nbrs)
        except:
            continue
        perm_t = torch.where (perm_t[0] | perm_t[1])[0]
        perm_t1 = torch.where (perm_t1[0] | perm_t1[1])[0]
        m, n = torch.max(torch.max(graphs[t+1].edge_index, dim=1).values, torch.max(graphs[t].edge_index, dim=1).values) + 1
        dA_ei = torch.cat((graphs[t].edge_index[:, perm_t], graphs[t+1].edge_index[:, perm_t1]), dim=1)
        dA_ew = torch.cat((graphs[t].edge_weight[perm_t], -graphs[t+1].edge_weight[perm_t1]))
        # dA_ei = torch.cat((graphs[t].edge_index, graphs[t-1].edge_index), dim=1)
        # dA_ew = torch.cat((graphs[t].edge_weight, -graphs[t-1].edge_weight))
        if ((0 in dA_ei.shape) or (0 in dA_ew.shape)):
            dA[t-1] = torch.tensor(1e-10)
        else:
            dA_ei, dA_ew = coalesce (dA_ei, dA_ew, m=m, n=n)
            dA[t-1] = (torch.square(dA_ew)).sum()**0.5
    return dA

def khop_nbrs (edge_index, nodes, k=2):
    # neighbors = torch.tensor([], dtype=torch.int64, device=edge_index.device)
    neighbors = nodes
    for _ in range(k):
        if (nodes.numel() == 0):
            break
        inds = torch.searchsorted(edge_index[0], torch.stack((nodes, nodes + 1)))
        nodes = torch.cat([edge_index[1, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
        # inds = torch.searchsorted(edge_index[1], torch.stack((nodes, nodes + 1)))
        # nodes2 = torch.cat([edge_index[0, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
        # nodes = torch.cat((nodes1, nodes2))
        neighbors = torch.cat ((neighbors, nodes))
    return neighbors 

def iter_ts_norm (target_nds, embs, graphs, k=2):
    emb_diffs = []
    for t in range(len(embs)-1):
        nbrs = khop_nbrs(graphs[t].edge_index, torch.tensor([node for node in target_nds if node in graphs[t].edge_index], 
                                                            dtype=torch.int64, device=graphs[t].edge_index.device), k=k)
        nbrs = torch.unique(nbrs)
        try:
            emb_diffs.append(torch.norm(embs[t+1][nbrs,:] - embs[t][nbrs,:]))
        except:
            pass
    return torch.stack(emb_diffs)

def get_stability_K (target_nodes, embs, graphs, feats=False, k=2, orig_embs=None, orig_graphs=None):
    dz = iter_ts_norm(target_nodes, embs, graphs, k=k)
    # print ('dzp:', dz)
    dA = norm_diff_graphs(target_nodes, graphs, k=k, feats=feats) # incorrect as A considers all the nodes and not just the k-hop neighborhood of the target nodes.
    # print ('dAp:', dA)
    N_dz = torch.divide(dz, dA)
    max_Ndz = torch.max(N_dz)
    min_Ndz = torch.min(N_dz)
    if ((orig_graphs is not None) and (orig_embs is not None)):
        dz_0 = iter_ts_norm(target_nodes, orig_embs, orig_graphs, k=k)
        # print ('dz0:', dz_0)
        dA_0 = norm_diff_graphs(target_nodes, orig_graphs, k=k, feats=feats) # incorrect as A considers all the nodes and not just the k-hop neighborhood of the target nodes.
        # print ('dA0:', dA_0)
        N_dz_0 = torch.divide(dz_0, dA_0)
        N_dz = N_dz[N_dz_0 > 0]
        N_dz_0 = N_dz_0[N_dz_0 > 0]
        if (len(N_dz_0) == 0):
            return torch.tensor(np.nan)
        max_Ndz_0 = torch.max(N_dz_0)
        min_Ndz_0 = torch.min(N_dz_0)
        # return sum_Ndz/sum_Ndz_0
        return (max_Ndz - min_Ndz)/(max_Ndz_0 - min_Ndz_0)
    return (max_Ndz - min_Ndz)

def get_stability_E (target_nodes, embs, graphs, k=2, orig_embs=None, orig_graphs=None):
    dz = iter_ts_norm(target_nodes, embs, graphs, k=k)
    max_dz = torch.max(dz)
    min_dz = torch.min(dz)
    if ((orig_graphs is not None) and (orig_embs is not None)):
        dz_0 = iter_ts_norm(target_nodes, orig_embs, orig_graphs, k=k)
        dz = dz[dz_0 > 0]
        dz_0 = dz_0[dz_0 > 0]
        if (len(dz_0) == 0):
            return torch.tensor(np.nan)
        max_dz_0 = torch.max(dz_0)
        min_dz_0 = torch.min(dz_0)
        # return sum_Ndz/sum_Ndz_0
        return torch.abs((max_dz - min_dz)/(max_dz_0 - min_dz_0) - 1)
    return (max_dz - min_dz)

def get_dzDiff_range (target_nodes, embs, graphs, orig_embs, orig_graphs, k=2):
    dz = iter_ts_norm(target_nodes, embs, graphs, k=k)
    dz_0 = iter_ts_norm(target_nodes, orig_embs, orig_graphs, k=k)
    dz = dz[dz_0 > 0]
    dz_0 = dz_0[dz_0 > 0]
    if (len(dz_0) == 0):
        return torch.tensor(np.nan)
    max_dz_diff = torch.max(dz - dz_0)
    min_dz_diff = torch.min(dz - dz_0)
    # return sum_Ndz/sum_Ndz_0
    return torch.abs(max_dz_diff - min_dz_diff)
    # return (max_dz - min_dz)


def get_dz_norm (target_nodes, embs, graphs, k=2):
    embs = [emb.to(graph.edge_index.device) for graph, emb in zip(graphs, embs)]
    dz = iter_ts_norm(target_nodes, embs, graphs, k=k)
    return dz

def get_dznorm_ratio (target_nodes, embs, graphs, orig_embs, orig_graphs, k=2):
    dz = iter_ts_norm(target_nodes, embs, graphs, k=k)
    dz_0 = iter_ts_norm(target_nodes, orig_embs, orig_graphs, k=k)
    return torch.sum(dz)/torch.sum(dz_0)

