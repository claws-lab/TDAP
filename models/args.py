import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='Argparser for training dynamic graph models')

parser.add_argument('-task', type=str, default='node_classification', help='prediction task')
parser.add_argument('-dataset', type=str, default='radoslaw', help='')
parser.add_argument('-num_graphs', type=int, default=13, help='Number of graphs')
parser.add_argument('-context', type=int, default=12, help='number of historical context used for inference.')
parser.add_argument('-target_ts', type=int, default=12, help='target snapshot')
parser.add_argument('-undirected', action='store_true', help='if the graph is undirected or not. Default is directed.')
parser.add_argument('-featureless', action='store_true', help='featureless data or not')
parser.add_argument('-large_graph', action='store_true', help='large graph or not')
parser.add_argument('-dyn_feats', action='store_true', help='dynamic features')

parser.add_argument('-model_name', type=str, default=None, help='Architecture model name if not default')
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

parser.add_argument('-nepochs', type=int, default=20, help='number of epochs')
parser.add_argument('-learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('-neg_weight', type=float, default=0.01, help='negative class weight')
parser.add_argument('-neg_sample_size', type=int, default=10, help='negative sampling size')
parser.add_argument('-batch_size', type=int, default=128, help='batch size')
parser.add_argument('-data_sample', action='store_true', help='Sampling the edge index')
parser.add_argument('-sample_prop', type=float, default=0.6, help='Sampling the edge index proportion')

parser.add_argument('-min_time', type=int, default=4, help='minimum time')
parser.add_argument('-min_time_perc', type=float, default=0.5, help='minimum time percentage')

parser.add_argument('-device', type=str, default='cpu', help='cpu/cuda')

parser.add_argument('-to_save', action='store_true', help='to save the model or not')
parser.add_argument('-logging', action='store_true', help='to log or not')

parser.add_argument('-seed', type=int, default=None, help='random seed')

args, _ = parser.parse_known_args()

args.dys_residual = not (args.dys_noresidual)