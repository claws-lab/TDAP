import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='Argparser for training dynamic graph models')

parser.add_argument('-task', type=str, default='node_classification', help='prediction task')
parser.add_argument('-dataset', type=str, default='ethereum_phishing', help='')
parser.add_argument('-num_graphs', type=int, default=11, help='Number of graphs')
parser.add_argument('-context', type=int, default=10, help='number of historical context used for inference.')
parser.add_argument('-target_ts', type=int, default=10, help='target snapshot')
parser.add_argument('-undirected', action='store_true', help='if the graph is undirected or not. Default is directed.')
parser.add_argument('-featureless', action='store_true', help='featureless data or not')
parser.add_argument('-large_graph', action='store_true', help='large graph or not')
parser.add_argument('-dyn_feats', action='store_true', help='dynamic features')

parser.add_argument('-model_name', type=str, default="EvGCNO", help='Architecture model name if not default')
parser.add_argument('-emb_size', type=int, default=128, help='embedding size')
parser.add_argument('-decoder_sizes', type=int, default=[64, 64], nargs='+', help='decoder sizes')

parser.add_argument('-chebyK', type=int, default=5, help='chebyshev order K for GC-LSTM')
parser.add_argument('-dys_struc_head', type=int, nargs='+', default=[16,8,8], help='DySAT: spatial encoders attention heads in each GAT layer')
parser.add_argument('-dys_struc_layer', type=int, nargs='+', default=[128], help='DySAT: # units in each spatial GAT layer')
parser.add_argument('-dys_temp_head', type=int, nargs='+', default=[16], help='DySAT: temporal encoders attention heads in each GAT layer')
parser.add_argument('-dys_temp_layer', type=int, nargs='+', default=[128], help='DySAT: # units in each spatial GAT layer')
parser.add_argument('-dys_spa_drop', type=float, default=0.1, help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('-dys_temp_drop', type=float, default=0.5, help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('-dys_noresidual', action='store_true', help='Use residual')

parser.add_argument('-nepochs', type=int, default=5, help='number of epochs')
parser.add_argument('-learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('-neg_weight', type=float, default=0.01, help='negative class weight')
parser.add_argument('-neg_sample_size', type=int, default=10, help='negative sampling size')
parser.add_argument('-reg_lambda', type=float, default=None, help='regularization lambda')

parser.add_argument('-batch_size', type=int, default=512, help='batch size')
parser.add_argument('-data_sample', action='store_true', help='Sampling the edge index')
parser.add_argument('-sample_prop', type=float, default=0.6, help='Sampling the edge index proportion')
parser.add_argument('-patience', type=int, default=None, help='patience for early stopping')

parser.add_argument('-min_time', type=int, default=4, help='minimum time')
parser.add_argument('-min_time_perc', type=float, default=0.5, help='minimum time percentage')

parser.add_argument('-device', type=str, default='cuda:1', help='cpu/cuda')

parser.add_argument('-to_save', action='store_true', help='to save the model or not')
parser.add_argument('-logging', action='store_true', help='to log or not')

parser.add_argument('-seed', type=int, default=None, help='random seed')

args, _ = parser.parse_known_args()

args.patience = args.nepochs if (args.patience is None) else args.patience

args.dys_residual = not (args.dys_noresidual)

# 
# 

import torch_geometric_temporal as tgtemp
import torch
import sys
sys.path.append("..")
from utils import *
from dataset import Dataset
import numpy as np
from model_wrapper import DynGraphVictim
import os
import torch.nn.functional as F

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
dyn_feats = args.dyn_feats

if (args.seed is not None):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

data = Dataset(root='../data', name=dataset, context=context_size, task=task, num_graphs=num_graphs, dyn_feats=dyn_feats,
                   ntargets=1, featureless=featureless, directed=not(undirected), device='cpu') # if (large_graph) else device) 
if (task == "node_classification"):
    data.data_split(target_snapshot, train_p=0.6, val_p=0.2, test_p=0.2, random_sample_nc=False)
elif (task == "edge_classification"):
    data.data_split(target_snapshot, train_p=0.1, val_p=0.1, test_p=0.8)
elif (task == "link_prediction"):
    data.link_split(target_snapshot, val_p=0.3, test_p=0.4)
# 
if not dyn_feats:
    data.normalize_feats()

try:
    data.features = torch.Tensor(np.array(data.features.todense()))
except:
    data.features = torch.Tensor(data.features)

# data.features = sparse_mx_to_torch_sparse_tensor(data.features) #.to(device)
# print (data.features.shape)
data.to_tg_data(num_ts=num_graphs, island=True)

# 
# 


from torch_geometric_temporal.signal.dynamic_graph_static_signal import DynamicGraphStaticSignal
from torch_geometric_temporal.signal import temporal_signal_split
feature = data.graphs[0].x.detach().cpu().numpy()
edge_indices = [g.edge_index.detach().cpu().numpy() for g in data.graphs]
edge_weights = [g.edge_weight.detach().cpu().numpy() for g in data.graphs]

tgt_data = DynamicGraphStaticSignal(edge_indices=edge_indices, edge_weights=edge_weights, feature=feature, targets=[None for _ in data.graphs])

# shuffled_inds = np.random.permutation(len(data.train_y))
train_nodes = data.train_mask.nonzero()[0] #[shuffled_inds]
train_labels = data.train_y #[shuffled_inds]
val_nodes = data.val_mask.nonzero()[0]
val_labels = data.val_y
test_nodes = data.test_mask.nonzero()[0]
test_labels = data.test_y

# print (data.labels.sum()/len(data.labels), 
#        train_labels.sum()/len(train_labels), 
#        val_labels.sum()/len(val_labels), 
#        test_labels.sum()/len(test_labels)
# )
# 
# 
# 
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from dataset import NodeBiClassSampler, NodeClassSampler

nfeats = data.graphs[0].x.shape[1]
nnodes = data.graphs[0].x.shape[0]
nclasses = len(torch.unique(train_labels))

# model parameters
model_name = args.model_name
emb_size = args.emb_size
decoder_sizes = args.decoder_sizes

# training parameters
num_epochs = args.nepochs
lr = args.learning_rate
batch_size = args.batch_size
neg_sample_size = args.neg_sample_size
neg_weight = args.neg_weight
reg_lambda = args.reg_lambda

if nclasses == 2:
    loader = DataLoader(NodeBiClassSampler(train_nodes, train_labels, neg_sample_size=neg_sample_size), 
                        batch_size=batch_size, num_workers=1)
else:
    loader = DataLoader(NodeClassSampler(train_nodes, train_labels), batch_size=batch_size, num_workers=1)
                            
model = DynGraphVictim(task=task, model_name=model_name, num_graphs=num_graphs, nclasses=nclasses, historical_len=context_size, nfeats=nfeats, nnodes=nnodes, emb_size=emb_size, 
                decoder_sizes=decoder_sizes, chebyK=args.chebyK, dys_struc_head=args.dys_struc_head, dys_struc_layer=args.dys_struc_layer,
                dys_temp_head=args.dys_temp_head, dys_temp_layer=args.dys_temp_layer, dys_spa_drop=args.dys_spa_drop, 
                dys_temp_drop=args.dys_temp_drop, dys_residual=args.dys_residual, device=device, debugging=False).to(torch.double).to(device)


# ce_loss = CrossEntropyLoss()
from sklearn.utils import class_weight
class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels.numpy())
print (class_weights)
nll_loss = torch.nn.NLLLoss(weight=torch.tensor(class_weights, dtype=torch.float64).to(device))
bce_loss = BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 
# 
def posneg_loss (model, graphs_t, pos_nodes, neg_nodes):
    emb_t = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
    pos_score = model.predict(emb_t, pos_nodes)
    neg_score = model.predict(emb_t, neg_nodes)
    pos_loss = bce_loss(pos_score, torch.ones_like(pos_score))/pos_nodes.shape[0]
    neg_loss = bce_loss(neg_score, torch.zeros_like(neg_score))/pos_nodes.shape[0]
    loss = pos_loss + neg_weight*neg_loss
    return loss

def back_bceloss (model, graphs_t, pos_nodes, neg_nodes):
    optimizer.zero_grad()
    loss = posneg_loss (model, graphs_t, pos_nodes, neg_nodes)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.detach().cpu().numpy().item()

def class_nll_loss (model, graphs_t, nodes, labels, backward=True):
    emb_t = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
    probs = model.predict(emb_t, nodes)
    # print(probs.shape, labels.shape)
    # print (probs, labels)
    loss = nll_loss (probs, labels)
    if reg_lambda is not None:
        loss += reg_lambda * sum(p.abs().sum() for p in model.parameters())/sum(1 for _ in model.parameters())
    if backward:
        loss.backward(retain_graph=True)
        optimizer.step()
        del model
        optimizer.zero_grad()
    return loss #.detach().cpu().numpy().item()
# 
# 
# 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from copy import deepcopy 

train_losses, val_losses = [], []
# val_nodes = torch.tensor(val_nodes)
# test_nodes = torch.tensor(test_nodes)
t = tgt_data.snapshot_count

torch.autograd.set_detect_anomaly(True)

model.train()
best_loss_val, best_val_metric = 100, 0.0
early_stopping, patience = args.patience, args.patience
best_weights = None
for epoch in tqdm(range(num_epochs), position=0, desc='Outer'):
    local_context = max(0, t - context_size)
    graphs_t = DynamicGraphStaticSignal (edge_indices=tgt_data.edge_indices[local_context:t], 
                                         edge_weights=tgt_data.edge_weights[local_context:t], 
                                         feature=tgt_data.feature, 
                                         targets=tgt_data.targets[local_context:t])
    av_loss = 0
    for batch_item in tqdm(loader, position=1, desc='Inner'):
        if (nclasses == 2):
            pos_nodes, neg_nodes = batch_item
            optimizer.zero_grad()
            loss = posneg_loss (model, graphs_t, pos_nodes, neg_nodes)
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            nodes, labels = batch_item
            loss = class_nll_loss(model, graphs_t, nodes, labels.to(device))
        av_loss += loss.detach().cpu().numpy().item()
        # av_loss += back_bceloss (model, graphs_t, pos_nodes, neg_nodes)
    with torch.no_grad():
        emb = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
        train_pred = model.predict(emb, torch.tensor(train_nodes)).detach().cpu().numpy().squeeze()
        val_pred = model.predict(emb, torch.tensor(val_nodes)).detach().cpu().numpy().squeeze()
        test_pred = model.predict(emb, torch.tensor(test_nodes)).detach().cpu().numpy().squeeze()
        # print (np.histogram(preds))
        # print (np.histogram(train_pred))
        # print (np.histogram(val_pred))
        # print (np.histogram(test_pred))
        # print(classification_report(train_labels, (train_pred > 0.5).astype(int)))
        # print(classification_report(test_labels, (test_pred > 0.5).astype(int)))
        if (nclasses == 2):
            if args.logging:
                print(classification_report(val_labels, (val_pred > 0.5).astype(int)))
            train_metric = roc_auc_score (train_labels, train_pred)
            val_metric = roc_auc_score (val_labels, val_pred)
            test_metric = roc_auc_score (test_labels, test_pred)
            val_loss = posneg_loss(model, graphs_t, val_nodes[val_labels==1], val_nodes[val_labels==0])
        else:
            if args.logging:
                print (confusion_matrix(val_labels, val_pred.argmax(axis=-1)))
            train_metric = accuracy_score (train_labels, train_pred.argmax(axis=-1))
            val_metric = accuracy_score (val_labels, val_pred.argmax(axis=-1))
            test_metric = accuracy_score (test_labels, test_pred.argmax(axis=-1))
            val_loss = class_nll_loss (model, graphs_t, val_nodes, val_labels.to(device), backward=False)
    if (val_metric > best_val_metric):
        best_acc_val = val_metric
        best_weights = deepcopy(model.state_dict())
        patience = early_stopping
    else:
        patience -= 1
    if ((epoch > early_stopping) and (patience <= 0)):
        break
    train_loss = av_loss/len(loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss.cpu().numpy())
    if args.logging:
        print (epoch, train_loss, val_loss, train_metric, val_metric, test_metric)

if (patience > 0):
    model.load_state_dict(best_weights)

import matplotlib.pyplot as plt
plt.plot(range(num_epochs), train_losses, 'b-', label='train')
plt.plot(range(num_epochs), val_losses, 'r--', label='val')
plt.legend()

plt.savefig('{}/train_val_loss_{}.png'.format(dataset, PID))


model_file_name = "{}/{}/modelncNew_{}_{}_{}.pt".format(model_name, dataset, num_graphs, context_size, target_snapshot)

import os
if not (os.path.exists(os.path.dirname(model_file_name))):
    os.makedirs (os.path.dirname(model_file_name))

from sklearn.metrics import roc_auc_score
with torch.no_grad():
    emb = model.hist_forward (graphs_t[-1].x.double(), graphs_t.edge_indices, graphs_t.edge_weights)
    preds = model.predict(emb, torch.tensor(test_nodes)).detach().cpu().numpy().squeeze()
    np.save(model_file_name.replace('model', 'preds')[:-3] + ".npy", preds)
    print (PID, args)
    if (nclasses == 2):
        print(classification_report(train_labels, (train_pred > 0.5).astype(int)))
        print(classification_report(val_labels, (val_pred > 0.5).astype(int)))
        print(classification_report(test_labels, (test_pred > 0.5).astype(int)))
        print (test_pred)
        train_auc = roc_auc_score (train_labels, train_pred)
        val_auc = roc_auc_score (val_labels, val_pred)
        test_auc = roc_auc_score (test_labels, test_pred)
        print (PID, "AUC:", train_auc, val_auc, test_auc)
    else:
        print (confusion_matrix(train_labels, train_pred.argmax(axis=-1)))
        print (confusion_matrix(val_labels, val_pred.argmax(axis=-1)))
        print (confusion_matrix(test_labels, test_pred.argmax(axis=-1)))
        train_acc = accuracy_score (train_labels, train_pred.argmax(axis=-1))
        val_acc = accuracy_score (val_labels, val_pred.argmax(axis=-1))
        test_acc = accuracy_score (test_labels, test_pred.argmax(axis=-1))
        print (PID, "Accuracy:", train_acc, val_acc, test_acc)
    import pickle as pkl
    pkl.dump(args.__dict__, open(model_file_name[:-3] + ".pkl", "wb"))
    # print (PID, np.histogram(preds))
    # print (np.histogram(preds[data.test_y == 0]))
    # print (np.histogram(preds[data.test_y == 1]))

if (args.to_save):
    torch.save (model.state_dict(), model_file_name)