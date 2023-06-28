from matplotlib.pyplot import get
import numpy as np
import copy
import torch
from functools import reduce
from torch_sparse import coalesce
import random
from torch.autograd import Variable
import sys
from models.model_wrapper import get_gpu_info
import utils
import time
from torch.nn import functional as F


def gradient_disturb(gradients: torch.FloatTensor, historical_len, nnodes, ratio):
    max_gradient_norm = torch.max(torch.abs(gradients)).item()
    num_disturbance = int(historical_len * nnodes * 0.3)
    disturbance_value = ratio*max_gradient_norm * np.ones((num_disturbance, ))
    disturbance_value = disturbance_value.astype(np.float)
    disturbance_indices = random.sample(range(nnodes), num_disturbance)
    disturbance_value = torch.zeros_like(gradients).index_put_((torch.LongTensor([random.sample(range(historical_len), 1)
                                                                for _ in range(num_disturbance)]),
                                                                torch.LongTensor(disturbance_indices)),
                                                               torch.FloatTensor(disturbance_value))
    return gradients + disturbance_value

def get_loss(output, y, nclasses=2, loss_type='CE'):
    if loss_type == "CE":
        if nclasses == 2:
            bce_loss = torch.nn.BCEWithLogitsLoss()
            # loss = F.nll_loss(output, y)
            loss = bce_loss(output, y.float())
            # loss = bce_loss(torch.unsqueeze(output, 0), torch.unsqueeze(y, 0))
        else:
            # nll_loss = torch.nn.NLLLoss(weight=torch.tensor(class_weights, dtype=torch.float))
            loss = F.nll_loss (torch.unsqueeze(output, dim=0), torch.unsqueeze(y, dim=0))
    elif loss_type == "CW":
        onehot = utils.tensor2onehot(y)
        best_second_class = (output - 1000*onehot).argmax(1)
        margin = output[np.arange(len(output)), y] - \
                output[np.arange(len(output)), best_second_class]
        k = 0
        loss = -torch.clamp(margin, min=k).mean()
        # loss = torch.clamp(margin.sum()+50, min=k)
    return loss

# grads = torch.zeros(self.num_graphs, self.ntg_nodes, self.nnodes, 2).to(self.device)
def get_grads (model, graphs, target_idx, target_nds, target_y, loss_type, nclasses=2,
               S_ew=None, num_graphs=13, ntg_nodes=2, nnodes=100, device='cpu'):
    grads = torch.zeros(num_graphs, ntg_nodes, nnodes, 2).to(device)
    prob = model.predict(model(graphs), target_idx)
    loss = get_loss(prob, target_y.to(prob.device), nclasses=nclasses, loss_type=loss_type)
    for t in range(grads.shape[0]):
        if S_ew is None:
            grad = torch.autograd.grad(loss, graphs[t].edge_weight, retain_graph=True, allow_unused=True)[0].data
        else:
            grad = torch.autograd.grad(loss, S_ew[t], retain_graph=True, allow_unused=True)[0].data
        grad = grad.detach().to(device)
        for itg, tg_node in enumerate(target_nds):
            out_idx = graphs[t].edge_index[0] == tg_node
            in_idx = graphs[t].edge_index[1] == tg_node
            grads[t, itg, graphs[t].edge_index[1, out_idx], 0] = grad[out_idx]
            grads[t, itg, graphs[t].edge_index[0, in_idx], 1] = grad[in_idx]
    del model, grad, graphs
    return grads

class TGA ():
    def __init__(
        self,
        model=None,
        num_graphs=0,
        directed=True,
        ntg_nodes=None,
        nnodes=None,
        nclasses=2,
        loss_type='CE',
        attack_structure=True,
        targetted=True,
        device='cpu',
        args_device='cuda:0',
        large_graph=False):
        assert nnodes is not None, 'Please give nnodes='
        assert ntg_nodes is not None

        self.victim_model = model
        self.attack_structure = attack_structure
        self.device = device if not large_graph else 'cpu'
        self.args_device = args_device

        self.loss_type = loss_type
        self.directed = directed

        self.num_graphs = num_graphs
        self.ntg_nodes = ntg_nodes
        self.nnodes = nnodes

        self.nclasses = nclasses

        self.targetted = targetted

        self.large_graph = large_graph

    def one_step_attack (self, ts_gradient, inputs, ts_edge_index, attack_history, ts, mode=None):
        def is_to_modify(g, link):
            if g > 0 and link == 1:
                return 0
            elif g <= 0 and link == 0:
                return 1
            else:
                return -1
        orig_shape = ts_gradient.shape
        gradients = ts_gradient.flatten()
        inputs = inputs.clone() # already copied
        _, sorted_index = torch.sort(torch.abs(gradients), descending=True)
        history = []
        for idx in sorted_index:
            orig_idx = np.unravel_index(idx.cpu().numpy(), orig_shape)
            ei_idx = torch.tensor([orig_idx[0], orig_idx[1]] if orig_idx[-1] == 0 else [orig_idx[0], orig_idx[1]])
            ei_idx = ei_idx.to(ts_edge_index.device)
            inds0 = torch.searchsorted(ts_edge_index[0], torch.stack((ei_idx[0], ei_idx[0]+1)))
            link = int (ei_idx[1] in ts_edge_index[1, inds0[0]:inds0[1]])
            value = is_to_modify(gradients[idx], link)
            if not mode and value != -1:
                if (ts, orig_idx) not in attack_history: #and len(history) < 5:
                #if (ts, sorted_index[i], value) not in attack_history and len(history) < 5:
                    inputs[ts][orig_idx] = 1
                    history.append((ts, orig_idx))
                    break
            elif mode == 'add' and (ts, orig_idx) not in attack_history:  #and len(history) < 5:
                if link == 0:
                    inputs[ts][orig_idx] = 1
                    history.append((ts, orig_idx))
                    break
        return inputs, history

    def attack_budget (self, graphs, target_y, target_idx, budget, disturbance_ratio=0, attack_mode=None, threshold=0.0):
        self.avail_nodes = np.array([torch.max(g.edge_index).item() for g in graphs])+1
        if (target_idx.ndim == 2):
            self.target_nds = torch.tensor(target_idx).reshape(target_idx.shape[0]*target_idx.shape[1]).to(self.device)
        elif (target_idx.ndim == 1):
            self.target_nds = torch.tensor(target_idx).to(self.device)
        else:
            self.target_nds = torch.tensor([target_idx]).to(self.device)
        target_y = torch.tensor(target_y).to(self.device)

        self.constraint = "budget"
        constraint_fn = lambda x: x.sum() <= budget
        
        self.victim_model.train()
        S_ew = [Variable(graphs[t].edge_weight.float(), requires_grad=True) for t in range(self.num_graphs)]
        grad_graphs = copy.deepcopy(graphs)
        for t in range(len(grad_graphs)):
            grad_graphs[t].edge_weight = S_ew[t]
        original_prob = self.victim_model.predict(self.victim_model(grad_graphs), target_idx)
        masked_loss = self._loss(original_prob, target_y.to(original_prob.device))

        # iteration #0
        best_adv_example = torch.zeros(self.num_graphs, self.ntg_nodes, self.nnodes, 2).to(self.device)
        grads = self.get_grads(graphs, masked_loss, S_ew)
        if disturbance_ratio > 0:
            gradients = gradient_disturb(grads.squeeze(), len(graphs), self.nnodes, disturbance_ratio)
        else:
            gradients = grads.squeeze()
        
        total_iter = 0
        attack_history = []
        attacked_graphs = copy.deepcopy(graphs)
        while constraint_fn(best_adv_example):
            total_iter += 1
            if (total_iter % 100 == 0):
                print (total_iter, file=sys.stderr)
            ah, adv_examples, probs = [], [], []
            for ts in range(len(graphs)):
                _adv_examples, _h = self.one_step_attack(gradients[ts], best_adv_example, 
                                        attacked_graphs[ts].edge_index, attack_history, ts, mode=attack_mode)
                adv_examples.append(_adv_examples)
                modified_graphs = self.get_modified_graphs(graphs, _adv_examples)
                probs.append(self.victim_model.predict(self.victim_model(modified_graphs), target_idx).detach().cpu().numpy())
                ah.append(_h)
            probs = np.array(probs)
            assert (target_idx.ndim == 1)
            min_id = np.argmin(probs)
            best_adv_example = _adv_examples[min_id]
            attack_history.append(ah[min_id])
            if (torch.abs(((1 - target_y) - probs[min_id])) < threshold):
                break
            
            # adv_perb_vec = Variable(best_adv_example, requires_grad=True)
            attacked_graphs = self.get_modified_graphs(graphs, best_adv_example)
            prob = self.victim_model.predict(self.victim_model(attacked_graphs), target_idx)
            masked_loss = self._loss(prob, target_y.to(prob.device))
            grads = self.get_grads(attacked_graphs, masked_loss)
            if disturbance_ratio > 0:
                gradients = gradient_disturb(grads.squeeze(), len(graphs), self.nnodes, disturbance_ratio)
            else:
                gradients = grads.squeeze()
        
        self.best_adv_example = best_adv_example
        self.modified_graphs = attacked_graphs
    
    def attack_noise (self, graphs, target_y, target_idx, epsilon, epsilon1, disturbance_ratio=0, attack_mode=None, threshold=0.0, debug=False, iter_ub=1000000, time_ub=250):
        self.victim_model.model.debugging = debug
        self.avail_nodes = np.array([torch.max(g.edge_index).item() for g in graphs])+1
        if (target_idx.ndim == 2):
            self.target_nds = torch.tensor(target_idx).reshape(target_idx.shape[0]*target_idx.shape[1]).to(self.device)
        elif (target_idx.ndim == 1):
            self.target_nds = torch.tensor(target_idx).to(self.device)
        else:
            self.target_nds = torch.tensor([target_idx]).to(self.device)

        target_y = torch.tensor(target_y).to(self.device)

        self.constraint = "noise"
        assert ((epsilon > 0) and (epsilon < 1))
        dA = self.norm_diff_graphs(graphs)
        bdgt1 = torch.tensor([min(min(dA*epsilon), epsilon1)], device=dA.device)
        epsilons_t = torch.cat ((torch.tensor([bdgt1], device=dA.device), epsilon * dA))
        constraint_fn=lambda x: reduce(lambda y, t: (x[t].sum() <= epsilons_t[t]) and y, range(x.shape[0]), True)
        print (dA, epsilons_t)

        self.victim_model.eval()
        S_ew = [Variable(graphs[t].edge_weight.float(), requires_grad=True) for t in range(self.num_graphs)]
        grad_graphs = copy.deepcopy(graphs)
        for t in range(len(grad_graphs)):
            grad_graphs[t].edge_weight = S_ew[t]
        # original_prob = self.victim_model.predict(self.victim_model(grad_graphs), target_idx)
        # masked_loss = self._loss(original_prob, target_y.to(original_prob.device))
        try:
            grads = get_grads (copy.deepcopy(self.victim_model), grad_graphs, target_idx, self.target_nds, 
                                target_y, self.loss_type, nclasses=self.nclasses, S_ew=S_ew, num_graphs=self.num_graphs, 
                                ntg_nodes=self.ntg_nodes, nnodes=self.nnodes, device=self.device)
        except:
            grads = get_grads (self.victim_model, grad_graphs, target_idx, self.target_nds, 
                            target_y, self.loss_type, nclasses=self.nclasses, S_ew=S_ew, num_graphs=self.num_graphs, 
                            ntg_nodes=self.ntg_nodes, nnodes=self.nnodes, device=self.device)
        # grads = self.get_grads(graphs, masked_loss, S_ew)
        
        # iteration #0
        best_adv_example = torch.zeros(self.num_graphs, self.ntg_nodes, self.nnodes, 2).to(self.device)
        if disturbance_ratio > 0:
            gradients = gradient_disturb(grads.squeeze(), len(graphs), self.nnodes, disturbance_ratio)
        else:
            gradients = grads.squeeze()
        
        total_iter = 0
        attack_history = []
        attacked_graphs = copy.deepcopy(graphs)
        attk_start_time = time.time()
        while True:
            total_iter += 1
            if debug:
                start_time = time.time()
                print (total_iter)
            if (total_iter >= iter_ub) or (time.time() - attk_start_time > time_ub):
                break
            if (total_iter % 100 == 0):
                print (total_iter)
            ah, adv_examples, probs = [], [], []
            for ts in range(len(attacked_graphs)):
                if debug:
                    print (ts, probs)
                    print (get_gpu_info(self.victim_model.device))
                _adv_examples, _h = self.one_step_attack(gradients[ts], best_adv_example, 
                                        attacked_graphs[ts].edge_index, attack_history, ts, mode=attack_mode)
                adv_examples.append(_adv_examples)
                if debug:
                    print (get_gpu_info(self.victim_model.device))
                modified_graphs = self.get_modified_graphs(attacked_graphs, _adv_examples)
                probs.append(self.victim_model.predict(self.victim_model(modified_graphs), target_idx).detach().cpu().numpy())
                # torch.cuda.empty_cache()
                if debug:
                    print (get_gpu_info(self.victim_model.device))
                ah.append(_h)
            probs = np.array(probs)
            # assert (target_idx.ndim == 1)
            if self.nclasses == 2:
                sorted_idx = sorted(range(len(probs)), reverse=(target_y == 0), key=lambda i: probs[i])
            else:
                sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i][target_y])
            found_id = -1
            for prob_sid in sorted_idx:
                if (constraint_fn(adv_examples[prob_sid])):
                    best_adv_example = adv_examples[prob_sid]
                    attack_history.append(ah[prob_sid])
                    found_id = prob_sid
                    break
            if (self.nclasses == 2):
                if (found_id == -1) or (torch.abs(((1 - target_y) - probs[prob_sid])) < threshold):
                    break
            else:
                if (found_id == -1):
                    break
            
            # adv_perb_vec = Variable(best_adv_example, requires_grad=True)
            attacked_graphs = self.get_modified_graphs(graphs, best_adv_example)
            # prob = self.victim_model.predict(self.victim_model(attacked_graphs), target_idx)
            # masked_loss = self._loss(prob, target_y.to(prob.device))
            try:
                grads = get_grads (copy.deepcopy(self.victim_model), attacked_graphs, target_idx, self.target_nds, 
                                target_y, self.loss_type, nclasses=self.nclasses, num_graphs=self.num_graphs, ntg_nodes=self.ntg_nodes, 
                                nnodes=self.nnodes, device=self.device)
            except:
                grads = get_grads (self.victim_model, attacked_graphs, target_idx, self.target_nds, 
                                target_y, self.loss_type, nclasses=self.nclasses, num_graphs=self.num_graphs, ntg_nodes=self.ntg_nodes, 
                                nnodes=self.nnodes, device=self.device)
            # grads = self.get_grads(attacked_graphs, masked_loss)
            # torch.cuda.empty_cache()
            # grad = torch.autograd.grad(masked_loss, adv_perb_vec, retain_graph=False)[0].data
            if debug:
                print (get_gpu_info(self.victim_model.device))
            if disturbance_ratio > 0:
                gradients = gradient_disturb(grads.squeeze(), len(graphs), self.nnodes, disturbance_ratio)
            else:
                gradients = grads.squeeze()
            
            if debug:
                print(time.time() - start_time)
                print (get_gpu_info(self.victim_model.device))
        
        self.best_adv_example = best_adv_example
        self.modified_graphs = attacked_graphs
        
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
            onehot = utils.tensor2onehot(y)
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
        
    def modify_undirec_graph(self, perbs, t, tni, target_node, graph_ei, graph_ew):
        conn_inds = torch.where(graph_ei[0] == target_node)[0]
        conn_nodes = graph_ei[1, conn_inds]
        all_conns = torch.zeros(self.nnodes, dtype=int, device=self.device)
        all_conns[conn_nodes] = 1
        complem = 1 - 2*all_conns
        complem[target_node] = 0 # no self cycles allowed 
        complem[self.avail_nodes[t]:self.nnodes] = 0 # if that node doesn't exist (i.e. no edge exists to/from it, then perturbations would not be valid from/to this)
        jperb_vec = complem * perbs[t,tni,:]
        jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                   torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        jperb_edges = torch.cat((jperb_edges, jperb_edges[[1,0]]), dim=1)
        jperb_vec = torch.cat ((jperb_vec, jperb_vec), dim=0)
        perm = torch.abs(jperb_vec) == 1
        jperb_edges, jperb_vec = jperb_edges[:, perm], jperb_vec[perm]
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def modify_direc_graph(self, perbs, t, tni, target_node, graph_ei, graph_ew, direc): #, max_nnodes):
        conn_inds = torch.where(graph_ei[direc] == target_node)[0]
        conn_nodes = graph_ei[1-direc, conn_inds]
        all_conns = torch.zeros(self.nnodes, dtype=int, device=self.device)
        all_conns[conn_nodes] = 1
        complem = 1 - 2*all_conns
        complem[target_node] = 0 # no self cycles allowed 
        complem[self.avail_nodes[t]:self.nnodes] = 0 # if that node doesn't exist (i.e. no edge exists to/from it, then perturbations would not be valid from/to this)
        jperb_vec = complem * perbs[t,tni,:,direc]
        if direc:
            # incoming
            jperb_edges = torch.stack((torch.arange(self.nnodes, dtype=int, device=self.device), 
                                       torch.full((self.nnodes,), target_node, dtype=int, device=self.device)))
        else:
            # outgoing
            jperb_edges = torch.stack((torch.full((self.nnodes,), target_node, dtype=int, device=self.device), 
                                       torch.arange(self.nnodes, dtype=int, device=self.device)))
        perm = torch.abs(jperb_vec) == 1
        jperb_edges, jperb_vec = jperb_edges[:, perm], jperb_vec[perm]
        graph_ei = torch.cat ((graph_ei, jperb_edges), dim=1)
        graph_ew = torch.cat ((graph_ew, jperb_vec))
        graph_ei, graph_ew = coalesce (graph_ei, graph_ew, m=self.nnodes, n=self.nnodes)
        perm = graph_ew != 0
        graph_ei, graph_ew = graph_ei[:, perm], graph_ew[perm]
        return graph_ei, graph_ew

    def get_modified_graphs(self, graphs, perbs, prune_x=False, to_print=False):
        from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        from torch_geometric.data import Data
        # from torch_geometric.utils import from_scipy_sparse_matrix, sort_edge_index
        modified_graphs = []
        for t in range(self.num_graphs):    
            modified_edge_index = graphs[t].edge_index.clone().to(self.device)
            modified_edge_weight = graphs[t].edge_weight.clone().to(self.device)
            # max_nnodes = torch.max(modified_edge_index)
            for j, node in enumerate(self.target_nds):
                # simpler version using coalesce ---
                if (self.directed):
                    modified_edge_index, modified_edge_weight = self.modify_direc_graph(perbs, t, j, node, modified_edge_index, modified_edge_weight, 0)
                    modified_edge_index, modified_edge_weight = self.modify_direc_graph(perbs, t, j, node, modified_edge_index, modified_edge_weight, 1)
                else:
                    modified_edge_index, modified_edge_weight = self.modify_undirec_graph(perbs, t, j, node, modified_edge_index, modified_edge_weight)

            x = graphs[t].x.clone()
            modified_edge_weight = Variable (modified_edge_weight.float(), requires_grad = True)
            modified_graphs.append(Data(x=x, edge_index=modified_edge_index, edge_weight=modified_edge_weight))
        return modified_graphs
