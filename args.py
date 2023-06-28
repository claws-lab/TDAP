import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='Argparser for WBA on dynamic graphs')
parser.add_argument('-constraint', type=str, default='budget', help='')
parser.add_argument('-saved_model', type=str, default=None, help='saved model directory')
parser.add_argument('-model_name', type=str, default=None, help='Architecture model name if not default')

parser.add_argument('-k_anom', type=int, default=2, help='Anomalous score of k')

parser.add_argument('-victims', type=str, nargs='+', default=None, help='Multiple victim models')

parser.add_argument('-dataset', type=str, default='radoslaw', help='')
parser.add_argument('-featureless', action='store_true', help='featureless data or not')
parser.add_argument('-undirected', action='store_true', help='if the graph is undirected or not. Default is directed.')
parser.add_argument('-dyn_feats', action='store_true', help='dynamic features or not')

parser.add_argument('-task', type=str, default='node_classification', help='prediction task')
parser.add_argument('-context', type=int, default=2, help='number of historical context used for inference.')
parser.add_argument('-target_ts', type=int, default=3, help='target snapshot')
parser.add_argument('-num_graphs', type=int, default=10, help='Number of graphs')

parser.add_argument('-budget', type=int, default=None, help='number of modifications allowed')
parser.add_argument('-budget_perc', type=float, default=None, help='budget percentage')
parser.add_argument('-bp_tgspec', action='store_true', help='percentage specific to each target or over all targets')

parser.add_argument('-epsilon', type=float, default=None, help='epsilon')
parser.add_argument('-epsilon1', type=int, default=None, help='budget for the first time')

parser.add_argument('-large_graph', action='store_true', help='large graph or not')

parser.add_argument('-ntargets', type=int, default=1, help='number of targets')

parser.add_argument('-online', action='store_true', help='online attack or not')

parser.add_argument('-inits', type=str, default='uniform', help='')
parser.add_argument('-num_steps', type=int, default=100, help='number of iterations for the attack GD')
parser.add_argument('-khop', type=int, default=2, help='no. of neighbors to consider')
parser.add_argument('-dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('-loss_type', type=str, default='CE', help='CE or CW')
parser.add_argument('-lambda1', type=float, default=0.01, help='regularization parameter')
parser.add_argument('-method', type=str, default='pgd', help='pgd or prbcd')
parser.add_argument('-thresh_wt', type=float, default=0.0, help='threshold of weights')
parser.add_argument('-lr_init', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-use_optim', action='store_true', help='optimizer or not')

parser.add_argument('-sampling', type=str, default='rd', help='rd or td or such criteria')
parser.add_argument('-num_samples', type=int, default=100, help='number of samples from the test set')
parser.add_argument('-neg_sampling', action='store_true', help='negative sampling')

parser.add_argument('-seq_tg_attk', action='store_true', help='sequentially attack targets')
parser.add_argument('-seq_order', type=str, default='rd', help='sequential order of targets')

parser.add_argument('-nprocs', type=int, default=1, help='number of processors')
parser.add_argument('-device', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('-pll_devices', nargs='+', type=str, default=None, help='cuda devices')

parser.add_argument('-seed', type=int, default=123, help='random seed')

parser.add_argument('-tga_disturbance_ratio', type=float, default=0.0, help='tga_disturbance_ratio')
parser.add_argument('-tga_attack_mode', type=str, default=None, help='tga attack mode - add or None')
parser.add_argument('-tga_thresh_prob', type=float, default=0.0, help='tga_thresh_prob')
parser.add_argument('-tga_iter_ub', type=int, default=100000, help='tga_iter_ub')

parser.add_argument('-iga_steps', type=int, default=10, help='iga_steps')

parser.add_argument('-debug', action='store_true', help='debugging on/off')

parser.add_argument('-save_only', action='store_true', help='save perbs only')
parser.add_argument('-analyze_only', action='store_true', help='analyze perbs only')
parser.add_argument('-save_perbs_file', type=str, help='save perbs file name')


parser.add_argument('-nosa', action='store_true', help='no structural attention')
parser.add_argument('-nota', action='store_true', help='no temporal attention')

cmd_args, _ = parser.parse_known_args()

if (cmd_args.pll_devices is None):
    cmd_args.pll_devices = [cmd_args.device] * cmd_args.nprocs
# print(cmd_args)
