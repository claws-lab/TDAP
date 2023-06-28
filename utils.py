import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.sparse 

def clip_grad_norm (gradients, max_norm, norm_type=2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = gradients.device
    if norm_type == float('inf'):
        norms = [gradients.detach().abs().max().to(device)]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(gradients, norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    gradients.detach().mul_(clip_coef_clamped.to(device))
    return gradients

def num_conn_nodes (graphs, links, nnodes):
    link_degs = []
    adjs = []
    for graph in graphs:
        from torch_geometric.utils import to_scipy_sparse_matrix
        adjs.append(to_scipy_sparse_matrix(graph.edge_index, num_nodes=nnodes).tocsr())
    links = links.reshape((1, -1)) if (links.ndim != 2) else links
    for link in links:
        adj = adjs[0]
        next_links = adj[link[0]] + adj[:, link[0]].T + adj[link[1]] + adj[:, link[1]].T
        for adj in adjs[1:]:
            next_links += adj[link[0]] + adj[:, link[0]].T + adj[link[1]] + adj[:, link[1]].T
        link_degs.append(next_links.nnz)
    link_degs = np.array(link_degs)
    return np.sum(link_degs)

def deg_sum (graphs, links, nnodes):
    link_degs = []
    adjs = []
    for graph in graphs:
        from torch_geometric.utils import to_scipy_sparse_matrix
        adjs.append(to_scipy_sparse_matrix(graph.edge_index, num_nodes=nnodes).tocsr())
    for link in links:
        next_links = 0
        for adj in adjs:
            next_links += adj[link[0]].sum() + adj[:, link[0]].sum() + adj[link[1]].sum() + adj[:, link[1]].sum()
        link_degs.append(next_links)
    link_degs = np.array(link_degs)
    return np.sum(link_degs)

def inv_map (ids):
    inv_ids = np.zeros_like(ids)
    for i, x in enumerate(ids):
        inv_ids[x] = i
    return inv_ids

def deg_sorted_links(graphs, links, labels, nnodes, descending=True):
    link_degs = []
    adjs = []
    for graph in graphs:
        from torch_geometric.utils import to_scipy_sparse_matrix
        adjs.append(to_scipy_sparse_matrix(graph.edge_index, num_nodes=nnodes).tocsr())
    for link in links:
        next_links = 0
        for adj in adjs:
            next_links += adj[link[0]].sum() + adj[:, link[0]].sum() + adj[link[1]].sum() + adj[:, link[1]].sum()
        link_degs.append(next_links)
    link_degs = np.array(link_degs)
    deg_ids = sorted(np.arange(len(links)), key=lambda i: link_degs[i], reverse=descending)
    del adjs
    return deg_ids, links[deg_ids], labels[deg_ids]

def prob_sorted_links(graphs, model, links, labels, large_graph=False, descending=True):
    embs = model(graphs, idx_targets=links) if (large_graph) else model(graphs)
    probs = model.predict(embs, links)
    mid = len(links) // 2
    prob_ids = sorted(np.arange(len(links)), key=lambda i: probs[i] if (i < mid) else (1 - probs[i]), 
                        reverse=descending)
    return prob_ids, links[prob_ids], labels[prob_ids]

def ovlp_sorted_links(graphs, links, labels, nnodes, descending=True):
    def khop_nbrs (edge_index, nodes, khop=2):
        neighbors = nodes
        for _ in range(khop):
            if (nodes.numel() == 0):
                break
            inds = torch.searchsorted(edge_index[0], torch.stack((nodes, nodes + 1)))
            nodes = torch.cat([edge_index[1, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
            neighbors = torch.cat ((neighbors, nodes))
        return neighbors
    counts = np.zeros(nnodes, dtype=int)
    for graph in graphs:
        nodes, freq_nodes = torch.unique(khop_nbrs(graph.edge_index, torch.tensor(links.ravel(), device=graph.edge_index.device)), return_counts=True)
        counts[nodes.cpu().numpy()] += freq_nodes.cpu().numpy()
    count_links = np.zeros(len(links), dtype=int)
    for j, link in enumerate(links):
        count_links[j] = counts[link[0]] + counts[link[1]]
    prob_ids = sorted(np.arange(len(links)), key=lambda i: count_links[i], reverse=descending)
    return prob_ids, links[prob_ids], labels[prob_ids]

def get_gpu_info (device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    t, r, a, f = t // (1024 ** 2), r // (1024 ** 2), a // (1024 ** 2), f // (1024 ** 2)
    print ("Total: {}, Reserved: {}, Allocated: {}, Free: {}".format(t, r, a, f))

def is_directed(adj):
    directed = True
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            directed &= graph[i, j] == graph[j, i]
            if graph[i, j] != graph[j, i]:
                print('i: ', i)
                print('j: ', j)
    print("Directed: ", directed)

# adj_matrices is a list of sparse matrices
# adjs is a tensor of nT, nnodes, nnodes
# feats/features is just nnodes, nfeats (not changing wrt time)

def to_pyg_graphs(features, adjs, device, labels=None, num_ts=None, island=True):
    from torch_geometric.data import Data
    from deeprobust.graph.utils import to_scipy
    from torch_geometric.utils import from_scipy_sparse_matrix
    pyg_graphs = []
    # adjs is a list of scipy sparse matrices...
    num_ts = len(adjs) if num_ts is None else num_ts
    for t in range(num_ts):
        edge_index, edge_weight = from_scipy_sparse_matrix(adjs[t])
        max_node = adjs[t].shape[0] if (island) else torch.max(edge_index)
        # All features are forced to be of the same size -- [:(max_node+1)]
        if (features.ndim == 3):
            data = Data(x=features[t], edge_index=edge_index, edge_weight=edge_weight, y=labels).to(device)
        else:
            data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels).to(device)
        pyg_graphs.append(data)
    return pyg_graphs

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def timeAdjs_to_sparseTensor (adj_matrices, num_ts=None):
    nT = num_ts if (num_ts is not None) else len(adj_matrices)
    t_i, t_v = [], []
    for t in range (nT):
        adj = sparse_mx_to_torch_sparse_tensor(adj_matrices[t])
        i, v = adj._indices(), adj._values()
        t_i.append(torch.cat((torch.full ((1, i.shape[1]), t), i), dim=0))
        t_v.append(v)
    adjs = torch.sparse_coo_tensor(torch.cat(t_i, dim=1), torch.cat(t_v))
    return adjs

def slice_timesteps (adjs, num_ts):
    t_i, t_v = [], []
    for t in range (num_ts):
        adj = sparse_mx_to_torch_sparse_tensor(adjs[t])
        i, v = adj._indices(), adj._values()
        t_i.append(torch.cat((torch.full ((1, i.shape[1]), t), i), dim=0))
        t_v.append(v)
    adjs = torch.sparse_coo_tensor(torch.cat(t_i, dim=1), torch.cat(t_v))
    return adjs

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_adjs(adjs):
    """Normalize adjacency tensor matrix.
    """
    from deeprobust.graph.utils import normalize_adj_tensor
    t_i, t_v = [], []
    for t in range(len(adjs)):
        norm_adj = normalize_adj_tensor(adjs[t], sparse=True)
        i, v = norm_adj._indices(), norm_adj._values()
        t_i.append(torch.cat((torch.full ((1, i.shape[1]), t), i), dim=0))
        t_v.append(v)
    norm_adjs = torch.sparse_coo_tensor(torch.cat(t_i, dim=1), torch.cat(t_v))
    return norm_adjs


def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_evaluation_data(adjs, num_time_steps, dataset):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = num_time_steps - 2
    eval_path = "data/{}/eval_{}.npz".format(dataset, str(eval_idx))
    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
        next_adjs = adjs[eval_idx + 1]
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(prev_adjs, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6, directed=False):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    def degsum (n):
        from functools import reduce
        return reduce(lambda x, y: x + y[n, :].sum(), prev_adjs, 0)
    if (type(prev_adjs) == list):
        adj = prev_adjs[-1]
        rmax, cmax = 0, 0
        for adj in prev_adjs:
            adj = adj[adj.getnnz(1)>0][:,adj.getnnz(0)>0]
            rmax, cmax = max(rmax, adj.shape[0]), max(cmax, adj.shape[1])
    else:
        adj = prev_adjs
        adj_p = adj[adj.getnnz(1)>0][:,adj.getnnz(0)>0]
        rmax, cmax = adj_p.shape[0], adj_p.shape[1]
    nmax = max(rmax, cmax)
    # nmax = adj.shape[0]
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    # assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    if (directed):
        edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj, create_using=nx.DiGraph).edges())))
    else:
        edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))

    edges = []   # Constraint to restrict new links to existing nodes.
    # print (adj.shape)
    for e in edges_next:
        if e[0] < rmax and e[1] < cmax and degsum(e[0]) > 0 and degsum(e[1]) > 0:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b):
        return a in b
        #rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        #return np.any(rows_close)
    
    def ismember_all(a, b):
        print(type(a))
        print(type(b))
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    def tup_to_list(arr):
        return []

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = set()
    edges_all = set([(edge[0], edge[1]) for edge in edges_all])
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, nmax)
        idx_j = np.random.randint(0, nmax)
        if idx_i == idx_j:
            continue
        if (degsum(idx_i) == 0 and degsum(idx_j) == 0):
            continue
        if ismember((idx_i, idx_j), edges_all):
            continue
        if ismember((idx_j, idx_i), edges_all):
            continue
        if train_edges_false:
            if ismember((idx_j, idx_i), train_edges_false):
                continue
            if ismember((idx_i, idx_j), train_edges_false):
                continue
        train_edges_false.add((idx_i, idx_j))
    train_edges_false = [list(edge) for edge in train_edges_false]

    # Create test edges.
    test_edges_false = set()
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, nmax)
        idx_j = np.random.randint(0, nmax)
        if idx_i == idx_j:
            continue
        if (degsum(idx_i) == 0 and degsum(idx_j) == 0):
            continue
        if ismember((idx_i, idx_j), edges_all):
            continue
        if ismember((idx_j, idx_i), edges_all):
            continue
        if test_edges_false:
            if ismember((idx_j, idx_i), test_edges_false):
                continue
            if ismember((idx_i, idx_j), test_edges_false):
                continue
        test_edges_false.add((idx_i, idx_j))
    test_edges_false = [list(edge) for edge in test_edges_false]

    # Create val edges.
    val_edges_false = set()
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if (degsum(idx_i) == 0 and degsum(idx_j) == 0):
            continue
        if ismember((idx_i, idx_j), edges_all):
            continue
        if ismember((idx_j, idx_i), edges_all):
            continue

        if val_edges_false:
            if ismember((idx_j, idx_i), val_edges_false):
                continue
            if ismember((idx_i, idx_j), val_edges_false):
                continue
        val_edges_false.add((idx_i, idx_j))
    val_edges_false = [list(edge) for edge in val_edges_false]
    
    #assert ~ismember_all(test_edges_false, edges_all)
    #assert ~ismember_all(val_edges_false, edges_all)
    #assert ~ismember_all(val_edges, train_edges)
    #assert ~ismember_all(test_edges, train_edges)
    #assert ~ismember_all(val_edges, test_edges)
    
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false


def random_split (target_t, labels, train_p, val_p, test_p):
    num_labels, num_classes = labels.shape[0], np.unique(labels[0])
    num_train_per_class = int(train_p*num_labels/num_classes)
    num_val, num_test = int(val_p*num_labels), int(test_p*num_labels)
    train_mask = np.zeros(num_labels, dtype=bool)
    val_mask = np.zeros(num_labels, dtype=bool)
    test_mask = np.zeros(num_labels, dtype=bool)
    for c in range(num_classes):
        idx = (labels[target_t] == c).nonzero(as_tuple=False).view(-1)
        idx = idx[np.randperm(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[np.randperm(remaining.size(0))]
    
    val_mask[remaining[:num_val]] = True
    test_mask[remaining[num_val:num_val + num_test]] = True
    train_mask = np.where(train_mask)[0]
    val_mask = np.where(val_mask)[0]
    test_mask = np.where(test_mask)[0]
    return train_mask, val_mask, test_mask
