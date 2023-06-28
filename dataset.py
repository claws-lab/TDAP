import itertools
import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
from utils import *
import json
import pickle as pkl
import random

def get_onehot_degree (num_nodes, edge_outs):
    deg_vec = np.zeros(num_nodes, dtype=int)
    for node in range(num_nodes):
        deg_vec[node] = np.sum(edge_outs == node)
    # 
    min_deg = np.min(deg_vec)
    max_deg = np.max(deg_vec)
    # deg1hot = np.zeros((num_nodes, max_deg - min_deg + 1), dtype=bool)
    nodes, degs = [], [], 
    for node in range(num_nodes):
        nodes.append(node)
        degs.append(deg_vec[node] - min_deg)
    return sp.csr_matrix((np.ones_like(nodes), (nodes, degs)), shape=(num_nodes, max_deg-min_deg+1), dtype=np.float64)


def deg_sorted_links(graphs, links):
    link_degs = []
    for link in links:
        num_ext_links = 0
        for graph in graphs:
            num_ext_links += graph[link[0]].sum() + graph[link[1]].sum()
        link_degs.append(num_ext_links)
    link_degs = np.array(link_degs)
    return links[sorted(np.arange(len(links)), key=lambda i: link_degs[i], reverse=True)]

def deg_sorted_nodes(graphs, nodes):
    node_degs = []
    for node in nodes:
        num_ext_nodes = 0
        for graph in graphs:
            num_ext_nodes += graph[node].sum()
        node_degs.append(num_ext_nodes)
    node_degs = np.array(node_degs)
    return nodes[sorted(np.arange(len(nodes)), key=lambda i: node_degs[i], reverse=True)]


def random_combinations(elements, r, ncombis):
    from iteration_utilities import random_combination
    l = []
    for _ in range(ncombis):
        l.append(list(random_combination(elements, r)))
    return np.array(l)

def seq_combinations(elements, r, ncombis):
    target_sets = []
    for edges in itertools.combinations(elements, r):
        target_sets.append(np.vstack(edges))
        if (len(target_sets) == ncombis):
            break
    return np.array(target_sets)

class Dataset ():
    def __init__(self, name='', num_graphs=None, ntargets=1, context=None, root='data', seed=123, dyn_feats=None, task='node_classification', normalize_feat=False, normalize_adjs=False, sparse=False, featureless=False, directed=True, device='cpu'):
        self.data_dir = root
        self.dataset = name
        self.directed = directed
        self.task = task
        self.num_graphs = num_graphs
        self.ntargets = ntargets
        self.context = context
        self.load_graphs()
        self.device = device
        self.dyn_feats = dyn_feats
        self.seed = seed

        try:
            self.load_feats()
        except:
            self.features = np.random.rand(self.max_nodes, 10) #.float() # random features
            # self.features = get_onehot_degree(self.max_nodes, self.adjs[-1].nonzero()[0])
            # self.features = sp.identity(self.max_nodes).tocsr()
            # self.features = sp.identity(self.max_nodes).tocsr()
            np.save("{}/{}/features.npy".format(self.data_dir, self.dataset), self.features)
        if ("classification" in task):
            # this will load for each time and each corresponding graph
            # Thus - nT X nnodes
            self.load_labels(task)
            self.labels = torch.LongTensor(self.labels)

        if normalize_adjs:
            self.normalize_adjs()

        if normalize_feat:
            self.normalize_feats()

        if sparse:
            self.adjs = timeAdjs_to_sparseTensor(self.adjs, num_ts=None)
            self.features = sparse_mx_to_torch_sparse_tensor(self.features)
        else:
            # remain scipy sparse matrices
            # self.features = torch.FloatTensor(np.array(self.features.todense()))
            # self.adjs = torch.FloatTensor(self.adjs.todense())
            pass
    
    def normalize_feats (self):
        self.features = normalize_feature(self.features)

    def normalize_adjs (self):
        self.adjs = np.vectorize(normalize_adj)(self.adjs)

    def to_sparseTensor (self, num_ts=None):
        self.adjs = timeAdjs_to_sparseTensor(self.adjs, num_ts=num_ts)
        self.features = sparse_mx_to_torch_sparse_tensor(self.features)

    def to_tg_data (self, num_ts=None, island=True):
        self.graphs = to_pyg_graphs (self.features, self.adjs, self.device, num_ts=num_ts, island=island)
        del self.adjs, self.features

    def load_graphs (self, padding=False):
        """Load graph snapshots given the name of dataset"""
        # need to first set it up like this as a pickle
        # graphs = np.load("{}/{}/{}".format(self.data_dir, self.dataset, "graphs.npz"), encoding='latin', allow_pickle=True)['graph']
        # print("Loaded {} graphs ".format(len(graphs)))
        try:
            with open("{}/{}/graphs_{}".format(self.data_dir, self.dataset, self.num_graphs), "rb") as f:
                graphs = pkl.load(f)
        except:
            # currently some problem -> use get_data.py from inside data/
            try:
                graphs = pkl.load(open("{}/{}/graphs_{}.pkl".format(self.data_dir, self.dataset, self.num_graphs), "rb"))
            except:
                from preprocess import load_edge_list
                graphs = load_edge_list(self.data_dir, self.dataset, self.num_graphs)
        print("Loaded {} graphs ".format(len(graphs)))
        self.adjs = graphs
        # self.adjs = [nx.adjacency_matrix(g) for g in graphs]
        for adj in self.adjs:
            # ignoring the weight or multiple edges...
            adj.data = np.ones_like (adj.data)
        # do padding here only -
        # Assuming no node deletions..
        self.max_nodes = max([adj.shape[0] for adj in self.adjs])
        if (padding):
            padded_adjs = []
            for adj in self.adjs:
                adj = sp.coo_matrix(adj)
                # adj.data = adj
                inds = [adj.row != adj.col]
                padded_adjs.append(sp.csr_matrix((adj.data[inds], (adj.row[inds], adj.col[inds])), shape=(self.max_nodes, self.max_nodes)))
            self.adjs = padded_adjs
        self.adjs = self.adjs[:self.num_graphs]

    def load_feats(self):
        """ Load node attribute snapshots given the name of dataset (not used in experiments)"""
        if self.dyn_feats:
            self.features = np.load("{}/{}/{}".format(self.data_dir, self.dataset, "dyn_features.npy")) #, allow_pickle=True)['feats']
        else:
            self.features = np.load("{}/{}/{}".format(self.data_dir, self.dataset, "features.npy")) #, allow_pickle=True)['feats']
        print("Loaded {} X matrices ".format(len(self.features)))
        # if (type(self.features) == list):
        #     self.features = self.features[-1]

    def load_labels(self, task):
        self.labels = np.load("{}/{}/{}/{}".format(self.data_dir, self.dataset, task, "labels.npy")) #, allow_pickle=True)['labels']

    def data_split (self, target_time_step, train_p=0.1, val_p=0.1, test_p=0.8, sampling='rd', num_samples=None):
        self.target_t = target_time_step
        if (self.task == "link_prediction"):
            self.link_split (target_time_step, val_p=val_p, test_p=test_p)
        elif (self.task == "node_classification"):
            self.class_split(train_p=train_p, val_p=val_p, test_p=test_p, num_samples=num_samples, sampling=sampling)
        elif (self.task == "edge_classification"):
            self.class_split(train_p=train_p, val_p=val_p, test_p=test_p)
            # get the coords
            edges, _ = sparse_to_tuple(self.adjs[target_time_step])
            self.train_mask = edges[self.train_mask]
            self.val_mask = edges[self.val_mask]
            self.test_mask = edges[self.test_mask]

    def class_split (self, train_p=0.1, val_p=0.1, test_p=0.8, num_samples=None, sampling='rd', random_sample_nc=False):
        self.random_split(train_p, val_p, test_p, num_samples=num_samples, sampling=sampling, random_sample=random_sample_nc)
        self.train_mask = np.where(self.train_mask)[0]
        self.val_mask = np.where(self.val_mask)[0]
        self.test_mask = np.where(self.test_mask)[0]

        self.train_y = self.labels[self.train_mask]
        self.val_y = self.labels[self.val_mask]
        self.test_y = self.labels[self.test_mask]

    def random_split (self, train_p, val_p, test_p, num_samples=None, sampling='rd', random_sample=False):
        if random_sample:
            eval_path = "{}/{}/{}/evalrandeq_{}_{}_{}.npz".format(
                self.data_dir,
                self.dataset,
                self.task,
                str(self.num_graphs),
                str(val_p)[2:],
                str(test_p)[2:]
            )
        else:
            eval_path = "{}/{}/{}/evalrand_{}_{}_{}.npz".format(
                self.data_dir,
                self.dataset,
                self.task,
                str(self.num_graphs),
                str(val_p)[2:],
                str(test_p)[2:]
            )
        # print (eval_path)
        classes = np.unique(self.labels)
        # print ([(c, (self.labels==c).sum().numpy()) for c in classes])
        minclass = min([(c, (self.labels==c).sum().numpy()) for c in classes], key=lambda x: x[1])
        # num_train = int(train_p * self.labels.shape[0])
        # num_val, num_test = int(val_p * self.labels.shape[0]), int(test_p * self.labels.shape[0])
        # print (num_train, num_val, num_test)
        # print (minclass)
        try:
            os.makedirs("{}/{}/{}/".format(self.data_dir, self.dataset, self.task))
        except:
            pass
        try:
            self.train_mask, self.val_mask, self.test_mask = np.load(
                            eval_path, encoding='bytes', allow_pickle=True)['data']
        except:
            print("Generating and saving eval data ....")
            self.train_mask = np.zeros(self.labels.shape[0], dtype=bool)
            self.val_mask = np.zeros(self.labels.shape[0], dtype=bool)
            self.test_mask = np.zeros(self.labels.shape[0], dtype=bool)
            for c in classes:
                idx = (self.labels == c).detach().cpu().numpy().nonzero()[0]
                if random_sample and (c != minclass[0]):
                    idx = idx[np.random.choice(range(idx.shape[0]), size=minclass[1], replace=False)]
                else:
                    idx = idx[np.random.permutation(idx.shape[0])]
                ntrain, nval = int(train_p * idx.shape[0]), int(val_p * idx.shape[0])
                ntest = int(test_p * num_samples) if num_samples is not None else int(test_p * idx.shape[0])
                self.train_mask[idx[:ntrain]] = True
                self.val_mask[idx[ntrain:ntrain+nval]] = True
                self.test_mask[idx[ntrain+nval:ntrain+nval+ntest]] = True

            print ("Generated")
            np.savez(eval_path, data=np.array([self.train_mask, self.val_mask, self.test_mask]))
        
        if num_samples is not None:
            if sampling == 'rd':
                test_mask_smpld = np.zeros_like(self.test_mask)
                for c in classes:
                    test_cids = (self.labels == c) & (self.train_mask)
                    test_c_nodes = np.where(test_cids)[0]
                    c_sids = np.random.choice(len(test_c_nodes), num_samples, replace=False)
                    print (c, c_sids)
                    test_mask_smpld[test_c_nodes[c_sids]] = True
                self.test_mask = test_mask_smpld
            elif sampling == 'td':
                test_mask_smpld = np.zeros_like(self.test_mask)
                for c in classes:
                    test_cids = np.where(((self.labels == c) & (self.train_mask)))[0]
                    test_c_nodes = deg_sorted_nodes(self.adjs[:-1], test_cids)
                    c_sids = np.random.choice(len(test_c_nodes), num_samples, replace=False)
                    # print (c, deg_sorted_nodes(self.adjs[:-1], test_cids))
                    test_mask_smpld[test_c_nodes[:num_samples]] = True
                self.test_mask = test_mask_smpld


    def random_traineq_split (self, train_p, val_p, test_p):
        eval_path = "{}/{}/{}/eval_{}_{}_{}.npz".format(
            self.data_dir,
            self.dataset,
            self.task,
            str(self.num_graphs),
            str(val_p)[2:],
            str(test_p)[2:]
        )
        # print (eval_path)
        print (self.labels)
        classes = np.unique(self.labels)
        print (classes)
        num_train_per_class = int(train_p * self.labels.shape[0]/len(classes))
        print (num_train_per_class)
        num_val, num_test = int(val_p * self.labels.shape[0]), int(test_p * self.labels.shape[0])
        try:
            os.makedirs("{}/{}/{}/".format(self.data_dir, self.dataset, self.task))
        except:
            pass
        try:
            self.train_mask, self.val_mask, self.test_mask = np.load(
                            eval_path, encoding='bytes', allow_pickle=True)['data']
        except:
            print("Generating and saving eval data ....")
            self.train_mask = np.zeros(self.labels.shape[0], dtype=bool)
            self.val_mask = np.zeros(self.labels.shape[0], dtype=bool)
            self.test_mask = np.zeros(self.labels.shape[0], dtype=bool)
            for c in classes:
                idx = (self.labels == c).detach().cpu().numpy().nonzero()[0]
                idx = idx[np.random.permutation(idx.shape[0])[:num_train_per_class]]
                self.train_mask[idx] = True

            remaining = (~self.train_mask).nonzero()[0]
            remaining = remaining[np.random.permutation(remaining.shape[0])]
            self.val_mask[remaining[:num_val]] = True

            self.test_mask[remaining[num_val:num_val + num_test]] = True
            print ("Generated")
            np.savez(eval_path, data=np.array([self.train_mask, self.val_mask, self.test_mask]))
        
    def link_split(self, target_time_step, val_p=0.2, test_p=0.6, sampling='rd', num_samples=100):
        """ Load train/val/test examples to evaluate link prediction performance"""
        eval_path = "{}/{}/{}/eval_{}_{}_{}_{}.npz".format(
            self.data_dir,
            self.dataset,
            self.task,
            str(target_time_step),
            str(self.num_graphs),
            str(val_p)[2:],
            str(test_p)[2:]
        )
        print (eval_path)
        try:
            os.makedirs("{}/{}/{}/".format(self.data_dir, self.dataset, self.task))
        except:
            pass
        try:
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
            print("Loaded eval data")
        except:
            print("Generating and saving eval data ....")
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                create_data_splits(self.adjs[:target_time_step], self.adjs[target_time_step], val_mask_fraction=val_p, test_mask_fraction=test_p, directed=self.directed)
            print ("Generated")
            np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                            test_edges, test_edges_false]))
        
        self.train_mask = np.concatenate((train_edges, train_edges_false))
        self.train_y = np.concatenate((np.ones(len(train_edges), dtype=int), np.zeros(len(train_edges), dtype=int)))
        # self.val_mask = np.concatenate((val_edges, val_edges_false))
        # self.val_y = np.concatenate((np.ones(len(val_edges), dtype=int), np.zeros(len(val_edges), dtype=int)))
        self.test_mask = np.concatenate((test_edges, test_edges_false))
        self.test_y = np.concatenate((np.ones(len(test_edges), dtype=int), np.zeros(len(test_edges_false), dtype=int)))
        if (sampling == 'rd'):
            if (self.ntargets > 1):
                if self.seed == 123:
                    target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_rd{}_{}_{}_{}.npy".format(
                                        self.ntargets, self.num_graphs, num_samples, target_time_step)
                else:
                    target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_rd{}_{}_{}_{}_{}.npy".format(
                                        self.ntargets, self.num_graphs, num_samples, target_time_step, self.seed)
                try:
                    # raise Exception()
                    self.test_mask = np.load(target_path)
                    print(self.test_mask.shape[:2])
                    self.test_y = np.zeros(shape=self.test_mask.shape[:2], dtype=int)
                    self.test_y[:, :(self.test_y.shape[1]//2)] = 1
                except:    
                    ntrue_edges = self.ntargets - self.ntargets//2
                    nfalse_edges = self.ntargets//2
                    # import itertools
                    # iter_edges_true = itertools.combinations(test_edges, self.ntargets - self.ntargets//2)
                    # iter_edges_false = itertools.combinations(test_edges_false, self.ntargets//2)
                    from iteration_utilities import random_combination
                    target_sets, target_ys = [], []
                    for _ in range(num_samples):
                        target_set = np.vstack((list(random_combination(test_edges, ntrue_edges)), list(random_combination(test_edges_false, nfalse_edges))))
                        target_sets.append(target_set)
                        target_ys.append(np.concatenate((np.ones(ntrue_edges, dtype=int), np.zeros(nfalse_edges, dtype=int))))
                    self.test_mask = np.stack(target_sets)
                    self.test_y = np.stack(target_ys)
                    np.save(target_path, self.test_mask)
                    # test_edges, test_y = [], []
                    # for e_true, e_false in zip(iter_edges_true, iter_edges_false):
                    #     if (len(test_edges) >= num_samples):
                    #         break
                    #     test_edges.append(np.vstack((e_true, e_false)))
                    #     test_y.append(np.concatenate((np.ones(len(e_true), dtype=int), np.zeros(len(e_false), dtype=int))))
            else:
                if self.seed == 123:
                    target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_rd{}_{}_{}_{}.npy".format(
                                        self.ntargets, self.num_graphs, num_samples, target_time_step)
                else:
                    target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_rd{}_{}_{}_{}_{}.npy".format(
                                        self.ntargets, self.num_graphs, num_samples, target_time_step, self.seed)
                try:
                    self.test_mask = np.load(target_path)
                    self.test_y = np.zeros(shape=(self.test_mask.shape[0],), dtype=int)
                    self.test_y[:(self.test_y.shape[0]//2)] = 1
                    print ("Loaded", target_path)
                except: 
                    test_edges_ids = np.random.randint(len(test_edges), size=num_samples)
                    test_edges_false_ids = np.random.randint(len(test_edges), size=num_samples)
                    self.test_mask = np.concatenate((np.array(test_edges)[test_edges_ids], np.array(test_edges_false)[test_edges_false_ids]))
                    self.test_y = np.concatenate((np.ones(num_samples, dtype=int), np.zeros(num_samples, dtype=int)))
                    for j in range(num_samples):
                        assert (self.adjs[target_time_step][self.test_mask[j][0], self.test_mask[j][1]] == 1)
                        assert (self.adjs[target_time_step][self.test_mask[num_samples+j][0], self.test_mask[num_samples+j][1]] == 0)
                    np.save(target_path, self.test_mask)
                # target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_rd{}_{}_{}_{}.npy".format(self.ntargets, self.num_graphs, num_samples, target_time_step)
                # try:
                #     self.test_mask = np.load(target_path)
                #     self.test_y = np.concatenate((np.ones(num_samples, dtype=int), np.zeros(num_samples, dtype=int)))
                #     for j in range(num_samples):
                #         assert (self.adjs[target_time_step][test_edges[j][0], test_edges[j][1]] == 1)
                # except:
                #     self.test_mask = random.sample(test_edges, num_samples), random.sample(test_edges_false, num_samples))
                #     np.save(target_path, self.test_mask)
        elif (sampling == 'td'):
            if (self.ntargets > 1):
                target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_td{}_{}_{}_{}.npy".format(self.ntargets, self.num_graphs, num_samples, target_time_step)
                try:
                    self.test_mask = np.load(target_path)
                    print(self.test_mask.shape[:2])
                    self.test_y = np.zeros(shape=self.test_mask.shape[:2], dtype=int)
                    self.test_y[:, :(self.test_y.shape[1]//2)] = 1
                except:    
                    import itertools
                    ntrue_edges = self.ntargets - self.ntargets//2
                    nfalse_edges = self.ntargets//2
                    # import itertools
                    # iter_edges_true = itertools.combinations(test_edges, self.ntargets - self.ntargets//2)
                    # iter_edges_false = itertools.combinations(test_edges_false, self.ntargets//2)
                    test_edges = deg_sorted_links(self.adjs, np.array(test_edges))
                    test_edges_false = deg_sorted_links(self.adjs, np.array(test_edges_false))
                    target_sets, target_ys = [], []
                    true_combis = seq_combinations(test_edges, ntrue_edges, num_samples)
                    false_combis = seq_combinations(test_edges_false, nfalse_edges, num_samples)
                    print(true_combis.shape, false_combis.shape)
                    self.test_mask = np.concatenate((true_combis, false_combis), axis=1)
                    self.test_y = np.concatenate((np.ones(shape=(num_samples, ntrue_edges), dtype=int), 
                                                  np.zeros(shape=(num_samples, nfalse_edges), dtype=int)), axis=1)
                    # for _ in range(num_samples):
                    #     target_set = np.vstack((, seq_combinations(test_edges_false, nfalse_edges)))
                    #     target_sets.append(target_set)
                    #     target_ys.append(np.concatenate((np.ones(ntrue_edges, dtype=int), np.zeros(nfalse_edges, dtype=int))))
                    # self.test_mask = np.stack(target_sets)
                    # self.test_y = np.stack(target_ys)
                    np.save(target_path, self.test_mask)
                    # test_edges, test_y = [], []
                    # for e_true, e_false in zip(iter_edges_true, iter_edges_false):
                    #     if (len(test_edges) >= num_samples):
                    #         break
                    #     test_edges.append(np.vstack((e_true, e_false)))
                    #     test_y.append(np.concatenate((np.ones(len(e_true), dtype=int), np.zeros(len(e_false), dtype=int))))
            else:    
                target_path = '/'.join(eval_path.split("/")[:-1]) + "/targets_td{}_{}_{}_{}.npy".format(self.ntargets, self.num_graphs, num_samples, target_time_step)
                try:
                    self.test_mask = np.load(target_path)
                    self.test_y = np.zeros(shape=(self.test_mask.shape[0],), dtype=int)
                    self.test_y[:(self.test_y.shape[0]//2)] = 1
                except:    
                    import itertools
                    test_edges = deg_sorted_links(self.adjs[:-1], np.array(test_edges))
                    test_edges_false = deg_sorted_links(self.adjs[:-1], np.array(test_edges_false))
                    self.test_mask = np.concatenate((test_edges[:num_samples], test_edges_false[:num_samples]))
                    self.test_y = np.concatenate((np.ones(shape=(num_samples,), dtype=int), 
                                                  np.zeros(shape=(num_samples,), dtype=int)))
                    np.save(target_path, self.test_mask)
                