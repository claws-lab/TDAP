import itertools
import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
import sys
sys.path.append ("..")
from utils import *
import json
import pickle as pkl
import random
import torch

from torch.utils.data import Dataset

import numpy as np

def negative_sample (edge, edge_index, sample_size=10):
    s, t = edge
    direcs = []
    null_edges = []
    while len(null_edges) < sample_size:
        if (np.random.rand() < 0.5):
            null_node = np.random.randint(0, edge_index.max())
            if null_node == s:
                continue
            s_ids = np.searchsorted(edge_index[0], np.stack([s, s+1]))
            if ((s_ids[0] < len(edge_index[0]) and (null_node in edge_index[1, s_ids[0]:s_ids[1]]))):
                continue
            direcs.append(1)
            null_edges.append (np.array([s, null_node]))
        else:
            null_node = np.random.randint(0, edge_index.max())
            if null_node == t:
                continue
            nullnode_ids = np.searchsorted(edge_index[0], np.stack([null_node, null_node+1]))
            if ((nullnode_ids[0] < len(edge_index[0])) and (t in edge_index[1, nullnode_ids[0]:nullnode_ids[1]])):
                continue
            direcs.append(0)
            null_edges.append (np.array([null_node, t]))
    return np.stack(null_edges), direcs

def sample_nodePairs(snapshot):
    edge_index = snapshot.edge_index.cpu().numpy()
    for nedge in range(edge_index.shape[1]):
        edge = edge_index[:, nedge]
        neg_edges, _ = negative_sample (edge, edge_index)
        yield edge, neg_edges

class LinkPredSampler (Dataset):
    def __init__ (self, snapshot, data_sample=False, sample_prop=0.7, neg_sample_size=10):
        # Take a random subset of this rather
        self.edge_index = snapshot.edge_index.cpu().numpy()
        self.full_ei = self.edge_index.copy()
        if data_sample:
            nedges = self.edge_index.shape[1]
            data_sample_size = int (sample_prop * nedges)
            sampled_idx = np.random.randint(0, nedges, size=(data_sample_size,))
            self.edge_index = self.edge_index[:, sampled_idx]
        self.neg_sample_size = neg_sample_size

    def __len__ (self):
        return self.edge_index.shape[1]

    def get_all (self):
        return self.edge_index.transpose(0, 1)

    def __getitem__(self, id):
        edge = self.edge_index[:, id]
        neg_edges, _ = negative_sample (edge, self.full_ei, sample_size=self.neg_sample_size)
        return edge, neg_edges

class NodeBiClassSampler (Dataset):
    def __init__ (self, nodes, labels, neg_sample_size=10):
        # Take a random subset of this rather
        self.nodes, self.labels = nodes, labels
        self.nodes1 = np.random.permutation(self.nodes[self.labels == 1])
        self.nodes0 = np.random.permutation(self.nodes[self.labels == 0])
        self.neg_sample_size = neg_sample_size

    def __len__ (self):
        return self.nodes1.shape[0]

    def get_all (self):
        return self.nodes1

    def __getitem__(self, id):
        pos_node = self.nodes1[id]
        if (self.neg_sample_size > 1):
            neg_node = np.random.choice(self.nodes0, size=self.neg_sample_size, replace=False)
        else:
            neg_node = self.nodes0[id]
        return pos_node, neg_node


class NodeClassSampler (Dataset):
    def __init__ (self, nodes, labels):
        # Take a random subset of this rather
        self.nodes = nodes
        self.labels = labels

    def __len__ (self):
        return self.nodes.shape[0]

    def __getitem__(self, id):
        return self.nodes[id], self.labels[id]


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
    def __init__(self, name='', num_graphs=None, ntargets=1, context=None, root='data', dyn_feats=False, task='node_classification', normalize_feat=False, normalize_adjs=False, sparse=False, featureless=False, directed=True, device='cpu'):
        self.data_dir = root
        self.dataset = name
        self.directed = directed
        self.task = task
        self.num_graphs = num_graphs
        self.ntargets = ntargets
        self.context = context
        self.load_graphs()
        self.device = device
        self.labels = None
        self.dyn_feats = dyn_feats
        try:
            self.load_feats()
        except:
            # self.features = np.random.rand(self.max_nodes, 10) #.float() # random features
            self.features = get_onehot_degree(self.max_nodes, self.adjs[-1].nonzero()[0])
            # self.features = sp.identity(self.max_nodes).tocsr()
            np.save("{}/{}/features.npy".format(self.data_dir, self.dataset), self.features)
        # if (self.dataset == 'opsahl-ucsocial'):
        #     self.features = get_onehot_degree(self.max_nodes, self.adjs[-1].nonzero()[0])
            # self.features = np.random.rand(self.max_nodes, 32) #.float() # random features
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
        self.graphs = to_pyg_graphs (self.features, self.adjs, self.device, labels=self.labels, num_ts=num_ts, island=island)
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
        # print("Loaded {} graphs ".format(len(graphs)))
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
        # print("Loaded {} X matrices ".format(len(self.features)))
        # if (type(self.features) == list):
        #     self.features = self.features[-1]

    def load_labels(self, task):
        labels = np.load("{}/{}/{}/{}".format(self.data_dir, self.dataset, task, "labels.npy")) #, allow_pickle=True)['labels']
        classes_map = {c:i for i, c in enumerate(np.unique(labels))}
        self.labels = np.zeros_like(labels)
        for i in range(labels.shape[0]):
            self.labels[i] = classes_map[labels[i]]

    def data_split (self, target_time_step, train_p=0.1, val_p=0.1, test_p=0.8, random_sample_nc=False):
        if (self.task == "link_prediction"):
            self.link_split (target_time_step, val_p=val_p, test_p=test_p)
        elif (self.task == "node_classification"):
            self.class_split(train_p=train_p, val_p=val_p, test_p=test_p, random_sample_nc=random_sample_nc)
        elif (self.task == "edge_classification"):
            self.class_split(train_p=train_p, val_p=val_p, test_p=test_p)
            # get the coords
            edges, _ = sparse_to_tuple(self.adjs[target_time_step])
            self.train_mask = edges[self.train_mask]
            self.val_mask = edges[self.val_mask]
            self.test_mask = edges[self.test_mask]

    def class_split (self, train_p=0.1, val_p=0.1, test_p=0.8, random_sample_nc=False):
        self.random_split(train_p, val_p, test_p, random_sample=random_sample_nc)
        self.train_y = self.labels[self.train_mask]
        self.val_y = self.labels[self.val_mask]
        self.test_y = self.labels[self.test_mask]

    def random_split (self, train_p, val_p, test_p, random_sample=False):
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
                ntrain, nval, ntest = int(train_p * idx.shape[0]), int(val_p * idx.shape[0]), int(test_p * idx.shape[0])
                self.train_mask[idx[:ntrain]] = True
                self.val_mask[idx[ntrain:ntrain+nval]] = True
                self.test_mask[idx[ntrain+nval:]] = True

            print ("Generated")
            np.savez(eval_path, data=np.array([self.train_mask, self.val_mask, self.test_mask]))

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

    def link_split(self, target_time_step, val_p=0.2, test_p=0.6):
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
        # print (eval_path)
        try:
            os.makedirs("{}/{}/{}/".format(self.data_dir, self.dataset, self.task))
        except:
            pass
        try:
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
            # print("Loaded eval data")
        except:
            print("Generating and saving eval data ....")
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                create_data_splits(self.adjs[max(0, target_time_step-self.context):target_time_step], self.adjs[target_time_step], 
                                    val_mask_fraction=val_p, test_mask_fraction=test_p, directed=self.directed)
            print ("Generated")
            np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                            test_edges, test_edges_false]))
        
        self.train_mask = np.concatenate((train_edges, train_edges_false))
        self.train_y = np.concatenate((np.ones(len(train_edges), dtype=int), np.zeros(len(train_edges), dtype=int)))
        self.val_mask = np.concatenate((val_edges, val_edges_false))
        self.val_y = np.concatenate((np.ones(len(val_edges), dtype=int), np.zeros(len(val_edges), dtype=int)))
        self.test_mask = np.concatenate((test_edges, test_edges_false))
        self.test_y = np.concatenate((np.ones(len(test_edges), dtype=int), np.zeros(len(test_edges), dtype=int)))
