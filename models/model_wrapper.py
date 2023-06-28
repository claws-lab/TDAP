import torch
import torch.nn.functional as F
import torch_geometric_temporal as tgtemp
import numpy as np

def get_gpu_info (device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    t, r, a, f = t // (1024 ** 2), r // (1024 ** 2), a // (1024 ** 2), f // (1024 ** 2)
    return ("Total: {}, Reserved: {}, Allocated: {}, Free: {}".format(t, r, a, f))

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DynGraphVictim(torch.nn.Module):
    def __init__(self, 
        task='link_prediction', 
        model_name='GCLSTM', 
        num_graphs=13,
        historical_len=12,
        nfeats=100, 
        nnodes=100, 
        nclasses=None,
        emb_size=128, 
        decoder_sizes=[64, 64], 
        chebyK=5, 
        dropout=None,
        dys_struc_head=[],
        dys_struc_layer=[],
        dys_temp_head=[],
        dys_temp_layer=[],
        dys_residual=False,
        dys_spa_drop=0.4,
        dys_temp_drop=0.5,
        device='cuda:0',
        gcn_normalize=True,
        debugging=False,
        **kwargs
    ):
        super(DynGraphVictim, self).__init__()
        self.model_name = model_name
        self.device = device
        self.task = task
        self.debugging = debugging
        self.dropout = dropout
        if (model_name == 'GCLSTM'):
            self.encoder = tgtemp.GCLSTM(in_channels=nfeats, out_channels=emb_size, 
                                        K=chebyK, normalization='sym' if gcn_normalize else None, debugging=debugging)
            out_size = emb_size
        elif (model_name == 'EvGCNH'):
            self.encoder = tgtemp.EvolveGCNH(num_of_nodes=nnodes, in_channels=nfeats, normalize=gcn_normalize)
            out_size = nfeats #?
        elif (model_name == 'EvGCNO'):
            self.encoder = tgtemp.EvolveGCNO(in_channels=nfeats, normalize=gcn_normalize)
            out_size = nfeats #?
        elif (model_name == 'DySAT'):
            try:
                from DySAT_pytorch.models.model import DySAT
            except:
                from models.DySAT_pytorch.models.model import DySAT
            args = DotDict({'structural_head_config': dys_struc_head, 'structural_layer_config': dys_struc_layer, 
                            'temporal_head_config': dys_temp_head, 'temporal_layer_config': dys_temp_layer, 
                            'spatial_drop': dys_spa_drop, 'temporal_drop': dys_temp_drop, 'window': -1, 'residual': dys_residual})
            self.encoder = DySAT (args=args, num_features=nfeats, time_length=num_graphs, task=task, debugging=debugging)
            out_size = args.temporal_layer_config[-1]
        elif (model_name == 'DDNE'):
            try:
                from DDNE.model_ns import ddne
            except:
                from models.DDNE.model_ns import ddne
            self.encoder = ddne (enc_inp_size=emb_size, decoder_sizes=[], historical_len=historical_len, num_nodes=nnodes, task=task)
            out_size = 2 * emb_size * historical_len # [hl, hr]
            # because of historical_len being fixed, DDNE trains differently. 
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2*out_size if (task == 'link_prediction') else out_size, decoder_sizes[0]),
            torch.nn.ReLU()
        )
        for i in range(1, len(decoder_sizes)-1):
            self.decoder.add_module('dec_{}'.format(i), torch.nn.Sequential(
                torch.nn.Linear(decoder_sizes[i-1], decoder_sizes[i]), torch.nn.ReLU()))
        if (task == 'link_prediction'):
            self.decoder.add_module('dec_{}'.format(len(decoder_sizes)), torch.nn.Sequential(
                torch.nn.Linear(decoder_sizes[-1], 1), torch.nn.Sigmoid()))
        elif (task == 'node_classification'):
            self.nclasses = nclasses
            if nclasses == 2:
                self.decoder.add_module('dec_{}'.format(len(decoder_sizes)), torch.nn.Sequential(
                    torch.nn.Linear(decoder_sizes[-1], 1), torch.nn.Sigmoid()))
            else:
                self.decoder.add_module('dec_{}'.format(len(decoder_sizes)), 
                    torch.nn.Linear(decoder_sizes[-1], nclasses))
            self.decoder.apply(init_weights)


    def forward(self, x, edge_index, edge_weight):
        if (self.model_name == 'GCLSTM'):
            h, c = self.encoder(x, edge_index, edge_weight)
        elif (self.model_name == 'EvGCNH' or self.model_name == 'EvGCNO'):
            h = self.encoder(x, edge_index, edge_weight)
        elif (self.model_name == 'DySAT'):
            raise Exception ("DySAT does not forward a single graph")
        elif (self.model_name == 'DDNE'):
            raise NotImplemented
        return h

    def hist_forward(self, x, edge_indices, edge_weights, return_all_times=False):
        x = self.to_device(x)
        edge_indices, edge_weights = self.to_device(edge_indices), self.to_device(edge_weights)
        edge_weights = [ew.double() for ew in edge_weights]
        embs_t = []
        if self.debugging:
            print (get_gpu_info(self.device))
        if (self.model_name == 'GCLSTM'):
            # print (get_gpu_info(self.device))
            z, C = self.encoder (x, edge_indices[0], edge_weights[0])
            embs_t.append(z) #.cpu()) #z.cpu() if self.debugging else z)
            for ei, ew in zip (edge_indices[1:], edge_weights[1:]):
                # print (get_gpu_info(self.device))
                z, C = self.encoder (x, ei, ew, H=z, C=C)
                embs_t.append(z) #.cpu()) #z.cpu() if self.debugging else z)
                # if (self.debugging):
                #     print (get_gpu_info(self.device))
        elif (self.model_name == 'EvGCNH' or self.model_name == 'EvGCNO'):
            z = x.clone()
            for ei, ew in zip (edge_indices, edge_weights):
                z = self.encoder(z, ei, ew)
                embs_t.append(z)
        elif (self.model_name == 'DySAT'):
            from torch_geometric.data import Data
            embs_t = self.encoder(x, edge_indices, edge_weights).transpose(0, 1)
            embs_t = [embs_t[t] for t in range(embs_t.shape[0])]
            z = embs_t[-1].squeeze() 
            if self.debugging:
                print (get_gpu_info(self.device))
        elif (self.model_name == 'DDNE'):
            from torch_geometric.data import Data
            graphs = [Data(x=x, edge_index=ei, edge_weight=ew) for ei, ew in zip(edge_indices, edge_weights)]
            return self.encoder(graphs, x_is_tgData=True)
        return z if (not (return_all_times)) else torch.stack(embs_t)

    def predict (self, z, idx):
        if (self.task == "link_prediction"):
            h0 = z[torch.index_select(idx, -1, torch.tensor([0])).squeeze()].clone()
            h1 = z[torch.index_select(idx, -1, torch.tensor([1])).squeeze()].clone()
            h_e = torch.cat((h0, h1), dim=-1)
            return self.decoder(h_e)
        elif (self.task == "node_classification"):
            # h0 = z[torch.index_select(idx, -1, torch.tensor([0])).squeeze()].clone()
            # h1 = z[torch.index_select(idx, -1, torch.tensor([1])).squeeze()].clone()
            # h_e = torch.cat((h0, h1), dim=-1)
            # z = F.dropout(z, self.dropout, training=self.training)
            if (self.nclasses == 2):
                return self.decoder(z[idx])
            else:
                return F.log_softmax(self.decoder(z[idx]), dim=-1)
            # return self.decoder(z[idx])
    
    def to_device (self, x):
        def xtodev (x):
            if (type(x) == np.ndarray):
                return torch.tensor(x).to(self.device)
            else:
                return x.to(self.device)
        if (type (x) == list):
            return [xtodev(xi) for xi in x] 
        else:
            return xtodev(x)