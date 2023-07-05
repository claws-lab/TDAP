import torch
import numpy as np
import sys

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AttackModel(torch.nn.Module):
    def __init__ (self, kwargs):
        super(AttackModel, self).__init__()
        # 
        self.model_name = kwargs['model_name']
        self.down_task = kwargs['task']
        self.device = kwargs['device']
        self.target_snapshot = kwargs['target_ts']
        self.historical_len = kwargs['historical_len']
        self.nega_sampling = kwargs['neg_sampling']
        # self.train_edges = kwargs['train_edges']
        # self.train_labels = kwargs['train_labels']
        self.dyngraph = False
        if (kwargs['model_name'] in ['GCLSTM', 'DySAT', 'EvGCNO', 'EvGCNH', 'DDNE']):
            from models.model_wrapper import DynGraphVictim
            self.dyngraph = True
            self.model = DynGraphVictim(**kwargs)
        self.model = self.model.to(self.device)

    def forward (self, graphs, return_all_times=True, idx_targets=None, print_log=False):
        if (self.dyngraph):
            return self.model.hist_forward(graphs[-1].x, [g.edge_index for g in graphs], [g.edge_weight for g in graphs], return_all_times=return_all_times)
        
    def predict (self, z, idx):
        # task-specific: might need to change this as well. 
        single_input = idx.ndim != 2
        idx = idx.reshape((1, *idx.shape)) if (idx.ndim != 2) else idx
        if (self.dyngraph):
            pred = self.model.predict(z[-1].to(self.device), torch.tensor(idx)) if (z.ndim == 3) else self.model.predict(z.to(self.device), torch.tensor(idx))
            return pred[0] if (single_input) else pred