import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
# from deeprobust.graph.global_attack import BaseAttack
from torch_sparse import coalesce
import sys
import time 

class PRBCD():
    """PGD attack for graph data.
    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    Examples
    --------
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import DPGD
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, y = data.adj, data.features, data.y
    >>> adj, features, y = preprocess(adj, features, y, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=y.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, y, idx_train)
    >>> # Setup Attack Model
    >>> model = DPGD(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, y, idx_train, n_perturbations=10)
    >>> modified_adjs = model.modified_adjs
    """

    def __init__(self,
                 model=None,
                 num_graphs: int = 0,
                 targetted: bool =True,
                 khop: int = 2,
                 directed: bool = True,
                 ntg_nodes: int = None,
                 nnodes: int = None, 
                 loss_type: str ='CE',
                 inits: str = 'uniform',
                 keep_heuristic: str = 'WeightOnly',
                 lr_factor: float = 100,
                 display_step: int = 20,
                 epochs: int = 500,
                 fine_tune_epochs: int = 100,
                 search_space_size: int = 1_000_000,
                 with_early_stopping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-7,
                 max_final_samples: int = 20,
                 device: str = 'cpu',
                 **kwargs):

        self.victim_model = model
        self.num_graphs = num_graphs
        self.targetted = targetted
        self.loss_type = loss_type

        self.ntg_nodes = ntg_nodes
        self.directed = directed
        self.nnodes = nnodes
        self.khop = khop

        self.device = device

        self.keep_heuristic = keep_heuristic
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.epochs_resampling = epochs - fine_tune_epochs
        self.search_space_size = search_space_size
        self.with_early_stopping = with_early_stopping
        self.eps = eps
        self.do_synchronize = do_synchronize
        self.max_final_samples = max_final_samples

        assert nnodes is not None, 'Please give nnodes='
        assert ntg_nodes is not None
        if (targetted):
            if (self.directed):
                self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes, 2)).to(self.device) 
            else:
                self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, self.ntg_nodes, nnodes)).to(self.device) 
        else:
            # incorrect
            if (self.directed):
                self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, nnodes, nnodes, 2)).to(self.device) 
            else:
                self.adj_changes = Parameter(torch.FloatTensor(self.num_graphs, nnodes, nnodes)).to(self.device) 
        
        if inits == "uniform":
            torch.nn.init.uniform_ (self.adj_changes)
        elif inits == "zero":
            self.adj_changes.data.fill_ (0)
        elif inits == "normal":
            torch.nn.init.normal_ (self.adj_changes)
        elif inits == "xavier_uni":
            torch.nn.init.xavier_uniform_ (self.adj_changes)
        elif inits == "xavier_norm":
            torch.nn.init.xavier_normal_ (self.adj_changes)

    def attack(self, graphs, target_y, target_idx, n_perturbations, orig_embs=None, lambda1=1e-2, **kwargs):
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
        """
        self.avail_nodes = np.array([torch.max(g.edge_index).item() for g in graphs])+1
        if (target_idx.ndim == 2):
            self.target_nds = torch.tensor(target_idx).reshape(target_idx.shape[0]*target_idx.shape[1]).to(self.device)
        else:
            self.target_nds = torch.tensor(target_idx).to(self.device)
        # self._forbid_node_adds()
        # 
        # if (lambda1 != 0):
        dz_0 = self._iter_ts_norm(orig_embs, graphs)
        self.nonzero_inds = dz_0 > 0
        self.sum_dz_0 = torch.sum(dz_0[self.nonzero_inds])
        # print (dz_0, self.sum_dz_0)
        # 
        # ori_features is sparse Tensor 
        # victim_model would be specific to a task 
        self.victim_model.eval()

        # can we try this - I have tried sparse tensors but doesn't seem to work... :(
        # ori_adjs, ori_features, target_y = utils.to_tensor(ori_adjs, ori_features, target_y, device=self.device)
        target_y = torch.tensor(target_y).to(self.device)
        self.sample_random_block(n_perturbations)

        for t in tqdm(range(self.epochs), position=0, leave=True):
            # start_time = time.time()
            modified_graphs = self.get_modified_graphs(graphs)
            # print(time.time() - start_time)
            # adj_norm = utils.normalize_adj_tensor(modified_graphs)
            self.victim_model.train()
            embs = self.victim_model(modified_graphs)
            # print(time.time() - start_time)
            # loss = F.nll_loss(output[target_idx], target_y)
            # interpret based on the task inside the attack model
            output = self.victim_model.predict(embs, target_idx)
            # print(time.time() - start_time)
            loss = self._loss(output, target_y)
            if (lambda1 != 0):
                smooth_loss = self.smooth_loss(embs, modified_graphs, normalized_return=False)
                loss -= lambda1 * smooth_loss
            # sparse doesn't work for fuck's sake. AHHHHHHH. Convert this into edge_index and
            # rewrite the whole code...
            # print (self.adj_changes.shape)
            # print (torch.autograd.grad(loss, modified_graphs[0].edge_index))
            adj_grad = torch.autograd.grad(loss, self.adj_changes, allow_unused=True)[0]
            # print (adj_grad)

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
            
            self.projection(n_perturbations)
            # self._forbid_node_adds()
            if t < self.epochs_resampling - 1:
                self.resample_random_block(n_perturbations)

        self.random_sample(modified_graphs, target_y, target_idx, n_perturbations, orig_embs=orig_embs, lambda1=lambda1)
        # print (torch.where(self.adj_changes))
        self.modified_graphs = self.get_modified_graphs(graphs)
        self.victim_model.eval()


    def sample_random_block(self, n_perturbations: int = 0):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_random_block(self, n_perturbations: int):
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.search_space_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')



    def _forbid_node_adds (self):
        with torch.no_grad():
            for t in range(self.adj_changes.shape[0]):
                self.adj_changes[t, self.target_nds >= self.avail_nodes[t],:] = 0
                self.adj_changes[t,:,self.avail_nodes[t]:] = 0

    def random_sample(self, graphs, target_y, target_idx, n_perturbations, orig_embs=None, lambda1=1e-2):
        # check whether only one direction is picked or not..
        K = 20
        best_loss = -1000
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            nz_idx = np.stack(s.nonzero()[:-1]).T
            if (self.directed):
                for i in range(nz_idx.shape[0]):
                    j,k,l = nz_idx[i]
                    if (s[j,k,l][0] == s[j,k,l][1]):
                        s[j,k,l][np.random.randint(2)] = 0
                    else:
                        s[j,k,l][np.argmin(s[j,k,l])] = 0

            for i in range(K):
                sampled = np.random.binomial(1, s)
                # while True:
                    # sampled = np.random.binomial(1, s)
                    # if not (np.any((sampled[:,:,:,0] == sampled[:,:,:,1])^(sampled[:,:,:,0] == 1))):
                    #     break

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_graphs = self.get_modified_graphs(graphs)
                # adj_norm = utils.normalize_adj_tensor(modified_graphs)
                embs = self.victim_model(modified_graphs)
                output = self.victim_model.predict(embs, target_idx)
                loss = self._loss(output, target_y)
                # print (torch.where(self.adj_changes), output, target_y)
                if (lambda1 != 0):
                    smooth_loss = self.smooth_loss(embs, modified_graphs, normalized_return=False)
                    loss -= lambda1 * smooth_loss
                # loss = F.nll_loss(output[target_idx], target_y)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            # print (best_loss)
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, y):
        if self.loss_type == "CE":
            bce_loss = torch.nn.BCEWithLogitsLoss()
            # loss = F.nll_loss(output, y)
            loss = bce_loss(output, y.float())
            # loss = bce_loss(torch.unsqueeze(output, 0), torch.unsqueeze(y, 0))
        elif self.loss_type == "CW":
            onehot = utils.tensor2onehot(y)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), y] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def norm_diff (self, graphs):
        def rec_or_targets (ei, target_nds):
            if (target_nds.shape[0] == 1):
                return (ei == target_nds[0])
            else:
                return ((ei == target_nds[0]) | (rec_or_targets(ei, target_nds[1:])))
        # frobenius by default
        dA = torch.zeros(len(graphs) - 1, device=self.device)
        # frobenius wrt time as well
        for t in range(1, len(graphs)):
            # basically find the diff edge_index and edge_weight
            # perm_t = torch.where (rec_or_targets(graphs[t].edge_index, self.target_nds))[1]
            # perm_t1 = torch.where (rec_or_targets(graphs[t-1].edge_index, self.target_nds))[1]
            m, n = torch.max(torch.max(graphs[t-1].edge_index, dim=1).values, torch.max(graphs[t].edge_index, dim=1).values) + 1
            # dA_ei = torch.cat((graphs[t].edge_index[:, perm_t], graphs[t-1].edge_index[:, perm_t1]), dim=1)
            # dA_ew = torch.cat((graphs[t].edge_weight[perm_t], -graphs[t-1].edge_weight[perm_t1]))
            # if ((0 in dA_ei.shape) or (0 in dA_ew.shape)):
            #     dA[t-1] = 1e-10
            # else:
            dA_ei = torch.cat((graphs[t].edge_index, graphs[t-1].edge_index), dim=1)
            dA_ew = torch.cat((graphs[t].edge_weight, -graphs[t-1].edge_weight))
            dA_ei, dA_ew = coalesce (dA_ei, dA_ew, m=m, n=n)
            dA[t-1] = (torch.square(dA_ew)).sum()**0.5
        return dA 

    def _khop_nbrs (self, edge_index, nodes):
        # neighbors = torch.tensor([], dtype=torch.int64, device=edge_index.device)
        neighbors = nodes
        for _ in range(self.khop):
            if (nodes.numel() == 0):
                break
            inds = torch.searchsorted(edge_index[0], torch.stack((nodes, nodes + 1)))
            nodes = torch.cat([edge_index[1, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
            # inds = torch.searchsorted(edge_index[1], torch.stack((nodes, nodes + 1)))
            # nodes2 = torch.cat([edge_index[0, inds[0, n]:inds[1, n]] for n in range(len(nodes))])
            # nodes = torch.cat((nodes1, nodes2))
            neighbors = torch.cat ((neighbors, nodes))
        return neighbors 

    def _iter_ts_norm (self, embs, graphs):
        emb_diffs = []
        for t in range(len(embs)-1):
            # print ([node for node in self.target_nds if node in graphs[t].edge_index])
            nbrs = self._khop_nbrs(graphs[t].edge_index, torch.tensor([node for node in self.target_nds if node in graphs[t].edge_index], 
                                                                        dtype=torch.int64, device=self.device))
            nbrs = torch.unique(nbrs)
            try:
                # print (nbrs, torch.norm(embs[t+1][nbrs,:] - embs[t][nbrs,:]))
                emb_diffs.append(torch.norm(embs[t+1][nbrs,:] - embs[t][nbrs,:]))
            except:
                pass
        return torch.stack(emb_diffs)

    def smooth_loss (self, embs, graphs, normalized_return=False, to_print=False):
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
            # tdr_sum = torch.sum(torch.divide(dz, dz_0))
            # return tdr_sum/dz.shape[0]
        # if (to_print):
        #     with open ("out_log.txt", "a") as f:
        #         print ("Perb:", torch.stack ((dz, dA)), N_dz, file=f)
        #         print ("Orig:", torch.stack((dz_0, dA_0)), N_dz_0, file=f)

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
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
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def get_modified_graphs(self, graphs, prune_x=False, to_print=False):
        # Edge index and edge weight 
        # Input => graphs
        from torch_geometric.data import Data
        # from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        modified_graphs = []
        for t in range(self.num_graphs):    
            modified_edge_index = graphs[t].edge_index.clone()
            modified_edge_weight = graphs[t].edge_weight.clone()
            # max_nnodes = torch.max(modified_edge_index)
            for j, node in enumerate(self.target_nds):
                # simpler version using coalesce ---
                if (self.directed):
                    modified_edge_index, modified_edge_weight = self.modify_direc_graph(t, j, node, modified_edge_index, modified_edge_weight, 0)
                    modified_edge_index, modified_edge_weight = self.modify_direc_graph(t, j, node, modified_edge_index, modified_edge_weight, 1)
                else:
                    modified_edge_index, modified_edge_weight = self.modify_undirec_graph(t, j, node, modified_edge_index, modified_edge_weight)

                if (to_print):
                    from torch_geometric.utils import to_scipy_sparse_matrix
                    with torch.no_grad():
                        mod_adj = sp.csr_matrix(to_scipy_sparse_matrix(modified_edge_index, modified_edge_weight, num_nodes=143))
                        orig_adj = sp.csr_matrix(to_scipy_sparse_matrix(graphs[t].edge_index, graphs[t].edge_weight, num_nodes=143))
                        print (torch.nonzero(self.adj_changes[t, j, :, 0]), orig_adj[node].nonzero()[1], mod_adj[node].nonzero()[1])
                        print (torch.nonzero(self.adj_changes[t, j, :, 1]), orig_adj[:,node].nonzero()[0], mod_adj[:,node].nonzero()[0])
            
            x = graphs[t].x.clone()
            if (prune_x):
                print (t, x.shape)
                island_nodes = torch.zeros(x.shape[0], dtype=bool)
                for j in range(x.shape[0]):
                    if (j not in modified_edge_index):
                        island_nodes[j] = True
                x = x[~island_nodes,:]
                print (x.shape)
                
            modified_graphs.append(Data(x=x, edge_index=modified_edge_index, edge_weight=modified_edge_weight).to(self.device))
        return modified_graphs
 
    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        # print(a, b)
        old_ba = b - a
        while ((b-a) >= epsilon):
            # print (a, b, b-a)
            miu = (a+b)/2
            # Check if middle point is root
            # print (a, b, func(miu), func(a))
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
            if (old_ba == (b - a)):
                break
            old_ba = b - a
        # print("The value of root is : ","%.4f" % miu)
        return miu
        