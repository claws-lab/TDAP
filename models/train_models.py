import torch_geometric_temporal as tgtemp
import torch
import sys
sys.path.append("..")
from utils import *
from dataset import Dataset
import numpy as np
from model_wrapper import DynGraphVictim
from args import args
import os

def get_gpu_info (device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    t, r, a, f = t // (1024 ** 2), r // (1024 ** 2), a // (1024 ** 2), f // (1024 ** 2)
    return ("Total: {}, Reserved: {}, Allocated: {}, Free: {}".format(t, r, a, f))

PID = os.getpid()
dataset = args.dataset
task = args.task
num_graphs, context_size, target_snapshot = args.num_graphs, args.context, args.target_ts
device = args.device
featureless = args.featureless
undirected = args.undirected
large_graph = args.large_graph

if (args.seed is not None):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

data = Dataset(root='../data', name=dataset, context=context_size, task=task, num_graphs=num_graphs, 
                   ntargets=1, featureless=featureless, directed=not(undirected), device='cpu') # if (large_graph) else device) 
if (task == "node_classification"):
    data.data_split(target_snapshot, train_p=0.6, val_p=0.2, test_p=0.2)
elif (task == "edge_classification"):
    data.data_split(target_snapshot, train_p=0.1, val_p=0.1, test_p=0.8)
elif (task == "link_prediction"):
    data.link_split(target_snapshot, val_p=0.3, test_p=0.4)
# 
data.normalize_feats()
try:
    data.features = torch.Tensor(np.array(data.features.todense()))
except:
    data.features = torch.Tensor(data.features)

# data.features = sparse_mx_to_torch_sparse_tensor(data.features) #.to(device)
data.to_tg_data(num_ts=num_graphs, island=True)

# 
# 
from torch_geometric_temporal.signal.dynamic_graph_static_signal import DynamicGraphStaticSignal
from torch_geometric_temporal.signal import temporal_signal_split
feature = data.graphs[0].x.detach().cpu().numpy()
edge_indices = [g.edge_index.detach().cpu().numpy() for g in data.graphs]
edge_weights = [g.edge_weight.detach().cpu().numpy() for g in data.graphs]

tgt_data = DynamicGraphStaticSignal(edge_indices=edge_indices, edge_weights=edge_weights, feature=feature, targets=[None for _ in data.graphs])
train_dataset, test_dataset = temporal_signal_split(tgt_data, train_ratio=target_snapshot/num_graphs)
# 
# 
# 
from tqdm import tqdm
from torch.nn.modules.loss import BCEWithLogitsLoss
from dataset import LinkPredSampler
from torch.utils.data import DataLoader

nfeats = train_dataset.feature.shape[1]
nnodes = train_dataset.feature.shape[0]

# model parameters
model_name = args.model_name
emb_size = args.emb_size
decoder_sizes = args.decoder_sizes

# training parameters
num_epochs = args.nepochs
lr = args.learning_rate
min_time = int(args.min_time_perc * len(train_dataset)) if (args.min_time_perc is not None) else args.min_time
neg_weight = args.neg_weight
neg_sample_size = args.neg_sample_size
batch_size = args.batch_size
data_sample = args.data_sample
sample_prop = args.sample_prop


sampler_loader = lambda snap: DataLoader(LinkPredSampler(snap, neg_sample_size=neg_sample_size, 
                                            data_sample=data_sample, sample_prop=sample_prop), 
                                        batch_size=batch_size, num_workers=4)

model = DynGraphVictim(task=task, model_name=model_name, num_graphs=num_graphs, historical_len=context_size, nfeats=nfeats, nnodes=nnodes, emb_size=emb_size, 
                decoder_sizes=decoder_sizes, chebyK=args.chebyK, dys_struc_head=args.dys_struc_head, dys_struc_layer=args.dys_struc_layer,
                dys_temp_head=args.dys_temp_head, dys_temp_layer=args.dys_temp_layer, dys_spa_drop=args.dys_spa_drop, 
                dys_temp_drop=args.dys_temp_drop, dys_residual=args.dys_residual, device=device, debugging=True).to(torch.double).to(device)

bce_loss = BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# NO EARLY STOPPING ~~~
# TO IMPLEMENT EARLY STOPPING...
torch.autograd.set_detect_anomaly(True)
for epoch in tqdm(range(num_epochs), position=0, desc='Outer'):
    for t, t_snapshot in enumerate(train_dataset):
        # print (epoch, t, get_gpu_info(device))
        local_context = max(0, t - context_size)
        graphs_t = DynamicGraphStaticSignal (edge_indices=train_dataset.edge_indices[local_context:t], 
                                             edge_weights=train_dataset.edge_weights[local_context:t], 
                                             feature=train_dataset.feature, 
                                             targets=train_dataset.targets[local_context:t])
        if (t > min_time):
            # static node features
            for pos_edges, neg_edges in tqdm(sampler_loader(t_snapshot), position=1, desc='Inner'):
                # print ("before forward:", get_gpu_info(device))
                emb_t = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
                # print ("after forward:", get_gpu_info(device))
                # print (torch.any(torch.isnan(emb_t)))
                pos_score = model.predict(emb_t, pos_edges)
                neg_score = model.predict(emb_t, neg_edges)
                pos_loss = bce_loss(pos_score, torch.ones_like(pos_score))/pos_edges.shape[0]
                neg_loss = bce_loss(neg_score, torch.zeros_like(neg_score))/pos_edges.shape[0]
                loss = pos_loss + neg_weight*neg_loss
                # print (get_gpu_info(device))
                if (args.logging):
                    from sklearn.metrics import roc_auc_score
                    with torch.no_grad():
                        emb = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
                        preds = model.predict(emb, torch.tensor(data.test_mask)).detach().cpu().numpy().squeeze()
                        # print (np.histogram(preds))
                        test_auc = roc_auc_score (data.test_y, preds)
                    print (epoch, t, loss, test_auc)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                # print (get_gpu_info(device))
                # print (torch.min(graphs_t.edge_weights[0].grad), torch.max(graphs_t.edge_weights[0].grad))
                del emb_t, pos_edges, neg_edges

model_file_name = "{}/{}/model_{}_{}_{}.pt".format(model_name, dataset, num_graphs, context_size, target_snapshot)

import os
if not (os.path.exists(os.path.dirname(model_file_name))):
    os.makedirs (os.path.dirname(model_file_name))

from sklearn.metrics import roc_auc_score
with torch.no_grad():
    emb = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
    preds = model.predict(emb, torch.tensor(data.test_mask)).detach().cpu().numpy().squeeze()
    np.save(model_file_name.replace('model', 'preds')[:-3] + ".npy", preds)
    test_auc = roc_auc_score (data.test_y, preds)
    print (PID, args)
    import pickle as pkl
    pkl.dump(args.__dict__, open(model_file_name[:-3] + ".pkl", "wb"))
    print (PID, "AUC on test data:", test_auc)
    print (PID, np.histogram(preds))
    # print (np.histogram(preds[data.test_y == 0]))
    # print (np.histogram(preds[data.test_y == 1]))

if (args.to_save):
    torch.save (model.state_dict(), model_file_name)