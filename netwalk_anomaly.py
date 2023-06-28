import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist
import torch
import copy
from torch_geometric.utils import coalesce

def kmeans_cluster(embs, train_indices, perb_edges, k=5, target_nds=None, edge_list=False):
    time_centroids = []
    for t in range(len(perb_edges)):
        src = embs[t, train_indices[:,0],:]
        dst = embs[t, train_indices[:,1],:]
        kmeans = KMeans(n_clusters=k)
        codes = np.multiply(src, dst)
        # Fitting the input data
        kmeans = kmeans.fit(codes)
        # Getting the cluster labels
        indices = kmeans.predict(codes)
        # Centroid values
        centroids = kmeans.cluster_centers_
        tbl = Counter(indices)
        n = list(tbl.values())
        time_centroids.append(centroids)
    # src = embs[test_indices[:,0],:]
    # dst = embs[test_indices[:,1],:]
    
    min_dists = []
    for t in range(len(perb_edges)):
        if edge_list: 
            test_src = embs[t, perb_edges[t][0]]
            test_dst = embs[t, perb_edges[t][1]]
        else:
            perb_target_ids, perb_nodes, perb_direc = np.where(perb_edges[t])
            perb_targets = target_nds[perb_target_ids] if target_nds is not None else perb_target_ids
            test_src = embs[t, perb_targets,:]
            test_dst = embs[t, perb_direc,:]

        test_codes = np.multiply(test_src, test_dst)
        # calculating distances for testing edge codes to centroids of clusters
        dist_center = cdist(test_codes, time_centroids[t])
        # assinging each testing edge code to nearest centroid
        min_dist = np.min(dist_center, 1)
        # sorting distances of testing edges to their nearst centroids
        scores = min_dist.argsort()
        scores = scores[::-1]

        # calculating distances for testing edge codes to centroids of clusters
        # dist_center_tr = cdist(codes, c)
        # min_dist_tr = np.min(dist_center_tr, 1)
        # max_dist_tr = np.max(min_dist_tr)
        # res = [1 if x > max_dist_tr else 0 for x in min_dist]
        #ab_score = np.sum(res)/(1e-10 + len(res))
        min_dists.append(min_dist)

    # min_dists = np.concatenate(min_dists)
    cat_min_dists = np.concatenate(min_dists)
    ab_score = np.sum(cat_min_dists) / (1e-10 + len(cat_min_dists))

    return min_dists, ab_score


def netwalk_clean_graph (mod_graphs, target_nds, embs, train_mask, k=2):
    all_edges = []
    for graph in mod_graphs:
        ei = graph.edge_index.cpu().numpy()
        e0, e1 = ei == target_nds[0]
        e0p, e1p = ei == target_nds[1]
        all_edges.append (ei[:, e0 | e1 | e0p | e1p])
    dists, _ = kmeans_cluster(embs, train_mask, all_edges, k=5, 
                            target_nds=target_nds, edge_list=True)
    new_graphs = []
    for dist, graph, edges in zip(dists, mod_graphs, all_edges):
        new_graph = graph
        ei = graph.edge_index
        ei_remove = torch.tensor (edges[:, dist > (dist.mean() + k * dist.std())], 
                    dtype=graph.edge_index.dtype, device=graph.edge_index.device)
        print (ei_remove)
        edge_attr = torch.cat ((torch.ones_like (ei[0]), -1 * torch.ones_like (ei_remove[0])))
        edge_index, edge_attr = coalesce (torch.cat((ei, ei_remove), dim=1), 
                                         edge_attr=edge_attr)
        new_graph.edge_index = edge_index[:, edge_attr == 1]
        new_graph.edge_weight = new_graph.edge_weight[edge_attr == 1]
        new_graphs.append(new_graph)
    return new_graphs