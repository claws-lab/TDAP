import numpy as np
import copy
import torch
from functools import reduce
from torch_sparse import coalesce
import random
from models.model_wrapper import get_gpu_info
import itertools

class RandAttack ():
    def __init__(
        self,
        num_graphs=0,
        directed=True,
        ntg_nodes=None,
        nnodes=None,
        nclasses=2,
        nfeats=2,
        attack_structure=True,
        attack_feature=False,
        targetted=True,
        device='cpu',
        args_device='cuda:0',
        large_graph=False):
        assert nnodes is not None, 'Please give nnodes='
        assert ntg_nodes is not None

        self.attack_structure = attack_structure
        self.device = device if not large_graph else 'cpu'
        self.args_device = args_device

        self.directed = directed

        self.num_graphs = num_graphs
        self.ntg_nodes = ntg_nodes
        self.nnodes = nnodes

        self.nclasses = nclasses

        self.targetted = targetted

        self.large_graph = large_graph

        self.attack_structure = attack_structure
        self.attack_feature = attack_feature

        self.adj_changes = {}
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            assert ntg_nodes is not None
            if (targetted):
                if (self.directed):
                    self.adj_changes['struc'] = torch.zeros(self.num_graphs, self.ntg_nodes, nnodes, 2).to(self.device)
                else:
                    self.adj_changes['struc'] = torch.zeros(self.num_graphs, self.ntg_nodes, nnodes).to(self.device) 
            else:
                # incorrect
                if (self.directed):
                    self.adj_changes['struc'] = torch.zeros(self.num_graphs, nnodes, nnodes, 2).to(self.device) 
                else:
                    self.adj_changes['struc'] = torch.zeros(self.num_graphs, nnodes, nnodes).to(self.device) 
        if attack_feature:
            assert nfeats is not None
            if (targetted):
                self.adj_changes['feat'] = torch.zeros(self.num_graphs, self.ntg_nodes, nfeats).to(self.device) 
            else:
                self.adj_changes['feat'] = torch.zeros(self.num_graphs, nnodes, nfeats).to(self.device) 

    def attack_noise (self, graphs, target_idx, epsilon, epsilon1, analyze_only=False):
        self.avail_nodes = np.array([torch.max(g.edge_index).item() for g in graphs])+1
        if (target_idx.ndim == 2):
            self.target_nds = torch.tensor(target_idx).reshape(target_idx.shape[0]*target_idx.shape[1]).to(self.device)
        elif (target_idx.ndim == 1):
            self.target_nds = torch.tensor(target_idx).to(self.device)
        else:
            self.target_nds = torch.tensor([target_idx]).to(self.device)

        if analyze_only:
            return

        self.constraint = "noise"
        assert ((epsilon > 0) and (epsilon < 1))
        epsilons_t, constants_t, n_perturbations = {}, {}, {}
        constraint_fn = {'struc': lambda x: True, 'feat': lambda x: True}
        if self.attack_structure:
            dA = self.norm_diff_graphs(graphs, attack='structure')
            bdgt1 = torch.tensor([min(min(dA*epsilon), epsilon1)], device=dA.device)
            constants_t['struc'] = torch.cat ((torch.tensor([bdgt1], device=dA.device), 2*dA))
            epsilons_t['struc'] = torch.cat ((torch.tensor([bdgt1], device=dA.device), epsilon * dA))
            constraint_fn['struc']=lambda x: reduce(lambda y, t: (x[t].sum() <= epsilons_t['struc'][t]) and y, range(x.shape[0]), True)
            n_perturbations['struc'] = epsilons_t
            # print (dA, epsilons_t)
        if self.attack_feature:
            dA = self.norm_diff_graphs(graphs, attack='feature')
            bdgt1 = torch.tensor([min(min(dA*epsilon), epsilon1)], device=dA.device)
            constants_t['feat'] = torch.cat ((torch.tensor([bdgt1], device=dA.device), 2*dA))
            epsilons_t['feat'] = torch.cat ((torch.tensor([bdgt1], device=dA.device), epsilon * dA))
            constraint_fn['feat']=lambda x: reduce(lambda y, t: (x[t].sum() <= epsilons_t['feat'][t]+0.01) and y, range(x.shape[0]), True)
            n_perturbations['feat'] = epsilons_t
            # print (dA, epsilons_t)

        for t in range(self.num_graphs):
            if self.attack_structure:
                # shapes = self.adj_changes['struc'][t].shape
                # all_inds = np.array(list(itertools.product(*[range(s) for s in shapes])))
                # np.random.randint
                # rand_inds = all_inds[np.random.choice(len(all_inds), size=int(epsilons_t['struc'][t].numpy()), replace=False)]
                # for ind in rand_inds:
                #     self.adj_changes['struc'][t][tuple(ind)] = 1
                while (self.adj_changes['struc'][t].sum() < epsilons_t['struc'][t]):
                    if self.targetted and self.directed:
                        self.adj_changes['struc'][t][np.random.randint(self.ntg_nodes), np.random.randint(self.nnodes), np.random.randint(2)] = 1
            if self.attack_feature:
                rand_perb = torch.rand(self.adj_changes['feat'][t].shape)
                rand_perb = rand_perb/rand_perb.sum() * epsilons_t['feat'][t]
                self.adj_changes['feat'][t] = rand_perb

        if self.attack_structure:
            self.modified_graphs = self.get_modified_graphs(graphs, attack='structure')
            if self.attack_feature:
                self.modified_graphs = self.get_modified_graphs(self.modified_graphs, attack='feature')
        elif self.attack_feature:
            self.modified_graphs = self.get_modified_graphs(graphs, attack='feature')

    
    def norm_diff_graphs (self, graphs, attack='structure'):
        def rec_or_targets (ei, nodes):
            if (nodes.shape[0] == 1):
                return (ei == nodes[0])
            else:
                return ((ei == nodes[0]) | (rec_or_targets(ei, nodes[1:])))
        # frobenius by default
        dA = torch.zeros(len(graphs) - 1) #, device=self.device)
        if attack == 'feature':
            for t in range(1, len(graphs)):
                # basically find the diff edge_index and edge_weight
                dA[t-1] = torch.abs(graphs[t].x[self.target_nds] - graphs[t-1].x[self.target_nds]).sum()
            return dA 
        elif attack == 'structure':
            # frobenius wrt time as well
            for t in range(1, len(graphs)):
                # basically find the diff edge_index and edge_weight
                perm_t = torch.where (rec_or_targets(graphs[t].edge_index, self.target_nds.to(graphs[t].edge_index.device)))[1]
                perm_t1 = torch.where (rec_or_targets(graphs[t-1].edge_index, self.target_nds.to(graphs[t].edge_index.device)))[1]
                m, n = torch.max(torch.max(graphs[t-1].edge_index, dim=1).values, torch.max(graphs[t].edge_index, dim=1).values) + 1
                dA_ei = torch.cat((graphs[t].edge_index[:, perm_t], graphs[t-1].edge_index[:, perm_t1]), dim=1)
                dA_ew = torch.cat((graphs[t].edge_weight[perm_t], -graphs[t-1].edge_weight[perm_t1]))
                if ((0 in dA_ei.shape) or (0 in dA_ew.shape)):
                    dA[t-1] = torch.tensor(0)
                else:
                    dA_ei, dA_ew = coalesce (dA_ei, dA_ew, m=m, n=n)
                    dA[t-1] = torch.abs(dA_ew).sum()
                    # dA[t-1] = (torch.square(dA_ew)).sum()**0.5
            return dA 

    def modify_undirec_graph(self, t, tni, target_node, graph_ei, graph_ew):
        conn_inds = torch.where(graph_ei[0] == target_node)[0]
        conn_nodes = graph_ei[1, conn_inds]
        all_conns = torch.zeros(self.nnodes, dtype=int, device=self.device)
        all_conns[conn_nodes] = 1
        complem = 1 - 2*all_conns
        complem[target_node] = 0 # no self cycles allowed 
        complem[self.avail_nodes[t]:self.nnodes] = 0 # if that node doesn't exist (i.e. no edge exists to/from it, then perturbations would not be valid from/to this)
        jperb_vec = complem * self.adj_changes['struc'][t,tni,:]
        jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                   torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        jperb_edges = torch.cat((jperb_edges, jperb_edges[[1,0]]), dim=1)
        jperb_vec = torch.cat ((jperb_vec, jperb_vec), dim=0)
        perm = torch.abs(jperb_vec) > 0
        jperb_edges, jperb_vec = jperb_edges[:, perm], jperb_vec[perm]
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def modify_direc_graph(self, t, tni, target_node, graph_ei, graph_ew, direc): #, max_nnodes):
        conn_inds = torch.where(graph_ei[direc] == target_node)[0]
        conn_nodes = graph_ei[1-direc, conn_inds]
        all_conns = torch.zeros(self.nnodes, dtype=int, device=self.device)
        all_conns[conn_nodes] = 1
        complem = 1 - 2*all_conns
        complem[target_node] = 0 # no self cycles allowed 
        complem[self.avail_nodes[t]:self.nnodes] = 0 # if that node doesn't exist (i.e. no edge exists to/from it, then perturbations would not be valid from/to this)
        jperb_vec = complem * self.adj_changes['struc'][t,tni,:,direc]
        if direc:
            # incoming
            jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                       torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        else:
            # outgoing
            jperb_edges = torch.stack((torch.full((self.nnodes,), target_node, dtype=int, device=self.device), 
                                       torch.arange(self.nnodes, dtype=int, device=self.device)))
        perm = torch.abs(jperb_vec) > 0
        jperb_edges, jperb_vec = jperb_edges[:, perm], jperb_vec[perm]
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def get_modified_graphs(self, graphs, attack='structure'):
        from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        from torch_geometric.data import Data
        # from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        modified_graphs = []
        for t in range(self.num_graphs):
            if attack == 'feature':
                modified_x = graphs[t].x.clone()
                for j, node in enumerate(self.target_nds):
                    # simpler version using coalesce ---
                    modified_x[node] = modified_x[node] + self.adj_changes['feat'][t, j, :]
                modified_graphs.append(Data(x=modified_x, edge_index=graphs[t].edge_index, edge_weight=graphs[t].edge_weight))
            elif attack == 'structure':
                modified_edge_index = graphs[t].edge_index.clone().to(self.device)
                modified_edge_weight = graphs[t].edge_weight.clone().to(self.device)
                # max_nnodes = torch.max(modified_edge_index)
                for j, node in enumerate(self.target_nds):
                    # simpler version using coalesce ---
                    if (self.directed):
                        modified_edge_index, modified_edge_weight = self.modify_direc_graph(t, j, node, modified_edge_index, modified_edge_weight, 0)
                        modified_edge_index, modified_edge_weight = self.modify_direc_graph(t, j, node, modified_edge_index, modified_edge_weight, 1)
                    else:
                        modified_edge_index, modified_edge_weight = self.modify_undirec_graph(t, j, node, modified_edge_index, modified_edge_weight)
                x = graphs[t].x.clone()                
                modified_graphs.append(Data(x=x, edge_index=modified_edge_index, edge_weight=modified_edge_weight))
        return modified_graphs
