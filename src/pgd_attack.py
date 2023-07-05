import numpy as np
import scipy.sparse as sp
import torch
from torch import device, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from torch_sparse import coalesce
import sys
import time 
from utils import get_gpu_info, clip_grad_norm, tensor2onehot
from functools import reduce
from torch.optim import Adam

class TDPGD():
    """PGD attack for discrete-time dynamic graph data.
    """

    def __init__(
        self,
        model=None,
        num_graphs=0,
        khop=2,
        large_graph=False,
        directed=True,
        ntg_nodes=None,
        nnodes=None,
        loss_type='CE',
        feature_shape=None,
        nclasses=2,
        attack_structure=True,
        attack_features=False,
        inits='uniform',
        targetted=True,
        thresh_wt=0,
        max_samples=20,
        device='cpu',
        args_device='cuda:0'):

        # super(TDPGD, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.victim_model = model
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        self.args_device = args_device
        # if model is not None:
        #     self.nclass = model.nclass
        #     self.nfeat = model.nfeat
        #     self.hidden_sizes = model.hidden_sizes

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_graphs = None
        self.directed = directed
        self.thresh_wt = thresh_wt

        self.num_graphs = num_graphs
        self.ntg_nodes = ntg_nodes
        self.nnodes = nnodes
        self.khop = khop
        self.large_graph = large_graph

        self.targetted = targetted

        self.max_samples = max_samples

        self.nclasses = nclasses

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            assert ntg_nodes is not None
            if (targetted):
                if (self.directed):
                    self.adj_changes = torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes, 2).to(self.device)
                    # self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes, 2)).to(self.device) 
                else:
                    self.adj_changes = torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes).to(self.device) 
                    # self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes)).to(self.device) 
            else:
                # incorrect
                if (self.directed):
                    self.adj_changes = torch.FloatTensor(self.num_graphs, nnodes, nnodes, 2).to(self.device) 
                    # self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, nnodes, nnodes, 2)).to(self.device) 
                else:
                    self.adj_changes = torch.FloatTensor(self.num_graphs, nnodes, nnodes).to(self.device) 
                    # self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, nnodes, nnodes)).to(self.device) 
            
            self.adj_changes.requires_grad = True
            if inits == "uniform":
                # self.adj_changes.data.fill_ (1.0)
                torch.nn.init.uniform_ (self.adj_changes)
            elif inits == "zeros":
                torch.nn.init.zeros_ (self.adj_changes)
            elif inits == "ones":
                torch.nn.init.ones_ (self.adj_changes)
            elif inits == "normal":
                torch.nn.init.normal_ (self.adj_changes)
            elif inits == "xavier_uni":
                torch.nn.init.xavier_uniform_ (self.adj_changes)
            elif inits == "xavier_norm":
                torch.nn.init.xavier_normal_ (self.adj_changes)

            # print (np.histogram(self.adj_changes.detach().cpu().numpy()))
        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, graphs, target_y, target_idx, constraint="budget", constr_vars=[], orig_embs=None, lambda1=1e-2, epochs=200, lr_init=10, use_optim=False, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        graphs:
            list of tg data objects
        # feats:
        #     torch tensor features (x)
        # adjs:
        #     scipy sparse csr matrices (adjs)
        #     list of sparse tensors 
        target_y :
            target y (task specific - nc, ec, lp)
        target_idx :
            target indices (task specific - nodes or edges)
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs
        constraint:
            noise/budget/ratio
        """
        self.avail_nodes = np.array([torch.max(g.edge_index).item() for g in graphs])+1
        if (target_idx.ndim == 2):
            self.target_nds = torch.tensor(target_idx).reshape(target_idx.shape[0]*target_idx.shape[1]).to(self.device)
        elif (target_idx.ndim == 1):
            self.target_nds = torch.tensor(target_idx).to(self.device)
        else:
            self.target_nds = torch.tensor([target_idx]).to(self.device)

        self.constraint = constraint
        if (constraint == "budget"):
            budget = constr_vars[0]
            constraint_fn = lambda x: x.sum() <= budget
            n_perturbations = budget
        elif (constraint == "noise"):
            epsilon, epsilon1 = constr_vars[0], constr_vars[1]
            assert ((epsilon > 0) and (epsilon < 1))
            dA = self.norm_diff_graphs(graphs)
            bdgt1 = torch.tensor([min(min(dA*epsilon), epsilon1)], device=dA.device)
            constants_t = torch.cat ((torch.tensor([bdgt1], device=dA.device), 2*dA))
            epsilons_t = torch.cat ((torch.tensor([bdgt1], device=dA.device), epsilon * dA))
            constraint_fn=lambda x: reduce(lambda y, t: (x[t].sum() <= epsilons_t[t]) and y, range(x.shape[0]), True)
            n_perturbations = epsilons_t
            print (dA, epsilons_t)

        # self._forbid_node_adds()
        dz_0 = self._iter_ts_norm(orig_embs, graphs)
        self.nonzero_inds = dz_0 > 0
        self.sum_dz_0 = torch.sum(dz_0[self.nonzero_inds])
        self.victim_model.eval()

        target_y = torch.tensor(target_y).to(self.device)
        if use_optim:
            optimizer = Adam([self.adj_changes], lr=lr_init)
        for t in tqdm(range(epochs), position=0, leave=True):
            modified_graphs = self.get_modified_graphs(graphs)
            # adj_norm = utils.normalize_adj_tensor(modified_graphs)
            self.victim_model.train()
            embs = self.victim_model(modified_graphs, idx_targets=target_idx) if (self.large_graph) else self.victim_model(modified_graphs)
            # embs = self.victim_model(graphs, idx_targets=target_idx) if (self.large_graph) else self.victim_model(graphs)
            output = self.victim_model.predict(embs, target_idx)
            loss = self._loss(output, target_y.to(output.device))
            # print (loss, torch.norm(self.adj_changes))
            if (lambda1 != 0):
                smooth_loss = self.smooth_loss(embs, modified_graphs, normalized_return=False)
                loss -= lambda1 * smooth_loss.to(loss.device)
            
            if use_optim:
                # torch.autograd.backward(loss, self.adj_changes)
                loss = - loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.adj_changes, 1.0)
                torch.nn.utils.clip_grad_value_(self.adj_changes, 1000)
                # print (torch.min(self.adj_changes.grad), torch.max(self.adj_changes.grad))
                optimizer.step()
                optimizer.zero_grad()
            else:
                adj_grad = torch.autograd.grad(loss, self.adj_changes, allow_unused=True)[0].to(self.device)
                adj_grad = torch.clamp (adj_grad, -1000, 1000)
                # torch.nn.utils.clip_grad_norm(self.adj_changes, 0)
                # clip_grad_norm (adj_grad, 100)
                # print (torch.min(adj_grad), torch.max(adj_grad))
                lr = lr_init / np.sqrt(t+1)
                if self.loss_type == 'CW':
                    lr = 0.01 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
            # print (torch.norm(embs-last_embs), output, loss)
            # print (np.histogram(adj_grad.cpu().numpy()))
            # print ("(", t, np.histogram(adj_grad.cpu().numpy()), ") ", end="| ", flush=True)
            # break
            
            # print (loss, adj_grad.sum())
            if (constraint == "budget"):
                self.projection_budget(budget)
            elif (constraint == "noise"):
                self.projection_noise(epsilons_t, constants_t)
            # self._forbid_node_adds()
        
        # print (output, loss, np.histogram(adj_grad.cpu().numpy()))
        self.random_sample(graphs, target_y, target_idx, n_perturbations, constraint_fn=constraint_fn, orig_embs=orig_embs, lambda1=lambda1)
        # print (torch.where(self.adj_changes))
        # print (self.adj_changes.sum(dim=list(range(1, self.adj_changes.ndim))))
        assert (constraint_fn(self.adj_changes))
        self.modified_graphs = self.get_modified_graphs(graphs)
        self.victim_model.eval()

    def _forbid_node_adds (self):
        with torch.no_grad():
            for t in range(self.adj_changes.shape[0]):
                self.adj_changes[t, self.target_nds >= self.avail_nodes[t],:] = 0
                self.adj_changes[t,:,self.avail_nodes[t]:] = 0

    @torch.no_grad()
    def random_sample(self, graphs, target_y, target_idx, n_perturbations, constraint_fn=lambda x: True, orig_embs=None, lambda1=1e-2):
        # check whether only one direction is picked or not..
        best_loss = -float ('Inf')
        s = self.adj_changes.cpu().detach().numpy()
        nz_idx = np.stack(s.nonzero()[:-1]).T
        if (self.directed):
            for i in range(nz_idx.shape[0]):
                j,k,l = nz_idx[i]
                if (s[j,k,l][0] == s[j,k,l][1]):
                    s[j,k,l][np.random.randint(2)] = 0
                else:
                    s[j,k,l][np.argmin(s[j,k,l])] = 0

        for i in range(self.max_samples):
            if best_loss == - float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled = np.zeros_like(s)
                if (self.constraint == 'budget'):
                    _nperb = int(np.ceil(n_perturbations/self.num_graphs))
                    total_perbs = n_perturbations
                elif (self.constraint == 'noise'):
                    total_perbs = n_perturbations.sum()
                # print (sampled.shape)
                for t in range(self.num_graphs-1, -1, -1):
                    if (self.constraint == 'noise'):
                        _nperb = n_perturbations[t]
                    try:
                        k = np.floor(min(_nperb, total_perbs)).long()
                    except:
                        k = np.floor(min(_nperb, total_perbs)).astype('int')
                    if (k != 0):
                        sampled[t][np.unravel_index(s[t].flatten().argsort()[-k:], s.shape[1:])] = 1
                    total_perbs -= k
            else:
                sampled = np.random.binomial(1, s)

            # print (i, sampled.sum(axis=tuple(range(1, sampled.ndim))), constraint_fn(sampled), file=sys.stderr)
            if not (constraint_fn(sampled)):
                continue
            self.adj_changes.data.copy_(torch.tensor(sampled))
            modified_graphs = self.get_modified_graphs(graphs)
            # adj_norm = utils.normalize_adj_tensor(modified_graphs)
            # print ([g.edge_weight[g.edge_weight<0] for g in graphs])
            # print ([g.edge_weight[g.edge_weight<0] for g in modified_graphs])
            embs = self.victim_model(modified_graphs, target_idx) if (self.large_graph) else self.victim_model(modified_graphs)
            output = self.victim_model.predict(embs, target_idx)
            loss = self._loss(output, target_y.to(output.device))
            # print (torch.where(self.adj_changes), output, target_y)
            if (lambda1 != 0):
                smooth_loss = self.smooth_loss(embs, modified_graphs, normalized_return=False)
                loss -= lambda1 * smooth_loss.to(output.device)
            # loss = F.nll_loss(output[target_idx], target_y)
            # print (embs, output, loss, best_loss)
            if best_loss < loss:
                best_loss = loss
                best_s = sampled
        self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, y):
        if self.loss_type == "CE":
            if self.nclasses == 2:
                bce_loss = torch.nn.BCEWithLogitsLoss()
                # loss = F.nll_loss(output, y)
                loss = bce_loss(output, y.float())
                # loss = bce_loss(torch.unsqueeze(output, 0), torch.unsqueeze(y, 0))
            else:
                # nll_loss = torch.nn.NLLLoss(weight=torch.tensor(class_weights, dtype=torch.float))
                loss = F.nll_loss (torch.unsqueeze(output, dim=0), torch.unsqueeze(y, dim=0))
        elif self.loss_type == "CW":
            onehot = tensor2onehot(y)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), y] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def norm_diff_graphs (self, graphs):
        def rec_or_targets (ei, nodes):
            if (nodes.shape[0] == 1):
                return (ei == nodes[0])
            else:
                return ((ei == nodes[0]) | (rec_or_targets(ei, nodes[1:])))
        # frobenius by default
        dA = torch.zeros(len(graphs) - 1) #, device=self.device)
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

    def _khop_nbrs (self, edge_index, nodes):
        # neighbors = torch.tensor([], dtype=torch.int64, device=edge_index.device)
        neighbors = nodes
        for _ in range(self.khop):
            if (nodes.numel() == 0):
                break
            inds = torch.searchsorted(edge_index[0], torch.stack((nodes, nodes + 1)))
            nodes = torch.cat([edge_index[1, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
            neighbors = torch.cat ((neighbors, nodes))
        return neighbors 

    def _iter_ts_norm (self, embs, graphs):
        emb_diffs = []
        for t in range(len(embs)-1):
            # print ([node for node in self.target_nds if node in graphs[t].edge_index])
            nbrs = self._khop_nbrs(graphs[t].edge_index, torch.tensor([node for node in self.target_nds.to(graphs[t].edge_index.device) 
                                                                        if node in graphs[t].edge_index], dtype=torch.int64,
                                                                        device=graphs[t].edge_index.device))
            nbrs = torch.unique(nbrs)
            try:
                # print (nbrs, torch.norm(embs[t+1][nbrs,:] - embs[t][nbrs,:]))
                emb_diffs.append(torch.norm(embs[t+1][nbrs,:] - embs[t][nbrs,:]))
            except:
                pass
        return torch.stack(emb_diffs)

    def smooth_loss (self, embs, graphs, normalized_return=False):
        embs = [emb.to(graph.edge_index.device) for graph, emb in zip(graphs, embs)]
        dz = self._iter_ts_norm(embs, graphs)
        # print (dz)
        # Don't need to calculate the value on orig graphs everytime. 
        if (normalized_return):
            assert (self.sum_dz_0 is not None)
            return torch.sum(dz[self.nonzero_inds])/self.sum_dz_0
        else:
            try:
                return torch.sum(dz[self.nonzero_inds])
            except:
                return torch.sum(dz)

    def projection_noise(self, epsilons_t, constants_t):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        P_01 = lambda x: torch.clamp(x, 0, 1)
        for t in range(self.adj_changes.shape[0]):
            at, ct, et = self.adj_changes[t], constants_t[t], epsilons_t[t]
            if (P_01(at).sum() > et):
                miu = TDPGD.bisection(at, (at - 1).min(), at.max(), et, epsilon=1e-5)
                self.adj_changes[t].data.copy_(P_01(at - miu))
            else:
                self.adj_changes[t].data.copy_(P_01(at))

    def projection_budget(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = TDPGD.bisection(self.adj_changes, left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))
        
    def modify_undirec_graph(self, t, tni, target_node, graph_ei, graph_ew):
        conn_inds = torch.where(graph_ei[0] == target_node)[0]
        conn_nodes = graph_ei[1, conn_inds]
        all_conns = torch.zeros(self.nnodes, dtype=int, device=self.device)
        all_conns[conn_nodes] = 1
        complem = 1 - 2*all_conns
        complem[target_node] = 0 # no self cycles allowed 
        complem[self.avail_nodes[t]:self.nnodes] = 0 # if that node doesn't exist (i.e. no edge exists to/from it, then perturbations would not be valid from/to this)
        jperb_vec = complem * self.adj_changes[t,tni,:]
        jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                   torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        jperb_edges = torch.cat((jperb_edges, jperb_edges[[1,0]]), dim=1)
        jperb_vec = torch.cat ((jperb_vec, jperb_vec), dim=0)
        perm = torch.abs(jperb_vec) > self.thresh_wt
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
        jperb_vec = complem * self.adj_changes[t,tni,:,direc]
        if direc:
            # incoming
            jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                       torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        else:
            # outgoing
            jperb_edges = torch.stack((torch.full((self.nnodes,), target_node, dtype=int, device=self.device), 
                                       torch.arange(self.nnodes, dtype=int, device=self.device)))
        perm = torch.abs(jperb_vec) > self.thresh_wt
        jperb_edges, jperb_vec = jperb_edges[:, perm], jperb_vec[perm]
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def get_modified_graphs(self, graphs, prune_x=False):
        from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        from torch_geometric.data import Data
        modified_graphs = []
        for t in range(self.num_graphs):    
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
            if (prune_x):
                print (t, x.shape)
                island_nodes = torch.zeros(x.shape[0], dtype=bool)
                for j in range(x.shape[0]):
                    if (j not in modified_edge_index):
                        island_nodes[j] = True
                x = x[~island_nodes,:]
                print (x.shape)
                
            modified_graphs.append(Data(x=x, edge_index=modified_edge_index, edge_weight=modified_edge_weight))
        return modified_graphs
    
    @staticmethod
    def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

        miu = a
        for i in range(int(iter_max)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= epsilon):
                break
        return miu
