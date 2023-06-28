from collections import defaultdict
import os
import datetime as dt
import networkx as nx
import pickle as pkl
import sys
import argparse
import math
import pandas as pd
import numpy as np

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='.', help='Dataset directory. Default: ''.''')
  parser.add_argument('--dataset', type=str, default='radoslaw', help='Dataset to load. Default: radoslaw')
  parser.add_argument('--task', type=str, default='link_prediction', help='Downstream Task. Default: link_prediction.')
  parser.add_argument('--historical_len', type=int, default=2, help='n_s in the paper, or lookback window. Default: 2')
  parser.add_argument('--num_graphs', type=int, default=7, help='number of timesteps. Default: 7')
  parser.add_argument('--interval', type=str, default=None, help='*{S|M|H|D|W}')
  parser.add_argument('--target_snapshot', type=int, default=3, help='the snapshot being targeted for attack. Default: 3')
  parser.add_argument('--analyze_only', action='store_true', help="Analyze only no saving")
  parser.add_argument('--discrete', action='store_true', help="Discrete graph input or not")
  parser.add_argument('--dyn_feats', action='store_true', help="Create structural dynamic features")
  parser.add_argument('--lb_time', type=int)
  parser.add_argument('--ub_time', type=int)
  return parser.parse_args()

def get_data_info (data_dir, dataset, lb_time=None, ub_time=None):
    files = os.listdir(data_dir + "/" + dataset)
    for datafile in files:
        if ((".out" in datafile) or ("out." in datafile) or (".edges" in datafile) or ("edges." in datafile)):
            min_t, max_t, max_node, n_edges = float('inf'), 0, 0, 0
            with open ("{}/{}/{}".format(data_dir, dataset, datafile), "r") as f:
                for line in f:
                    if ('%' not in line):
                        try:
                            u, v, w, t = [int(float(x)) for x in line[:-1].split()]
                        except:
                            u, v, w, t = [int(float(x)) for x in line[:-1].split(',')]
                        dt_time = dt.datetime.utcfromtimestamp(t)
                        if ((lb_time is not None) and (t < lb_time)) or ((ub_time is not None) and (t > ub_time)):
                            continue
                        max_node = max(max_node, u)
                        max_node = max(max_node, v)
                        max_t = max(t, max_t)
                        min_t = min(t, min_t)
                        n_edges += 1

            return datafile, max_t, min_t, max_node, n_edges


def get_discrete_graphs (data_dir, datafile, dataset, max_node, min_t, n_snapshots, time_interval, create_dyn_feats=False, discrete=False, lb_time=None, ub_time=None):
    def new_graph():
        graph = nx.DiGraph()
        graph.add_nodes_from(list(range(max_node)))
        return graph

    graphs, t_snapshot = [new_graph() for _ in range(n_snapshots)], 0
    out_weights = {t_snapshot: defaultdict(int) for t_snapshot in range(n_snapshots)}
    in_weights = {t_snapshot: defaultdict(int) for t_snapshot in range(n_snapshots)}
    out_degrees = {t_snapshot: defaultdict(int) for t_snapshot in range(n_snapshots)}
    in_degrees = {t_snapshot: defaultdict(int) for t_snapshot in range(n_snapshots)}
    with open ("{}/{}/{}".format(data_dir, dataset, datafile), "r") as f:
        for line in f:
            if ('%' not in line):
                try:
                    u, v, w, t = [float(x) for x in line[:-1].split()]
                except:
                    u, v, w, t = [float(x) for x in line[:-1].split(',')]
                # starts from 1 usually => but in networkx, would start from 0
                u, v = int(u - 1), int(v - 1)
                dt_time = dt.datetime.utcfromtimestamp(t)
                if ((lb_time is not None) and (t < lb_time)) or ((ub_time is not None) and (t > ub_time)):
                    continue
                t_snapshot = max(math.ceil((t - min_t)/time_interval) - 1, 0) if not discrete else t - 1
                out_weights[t_snapshot][u] += w
                in_weights[t_snapshot][v] += w
                out_degrees[t_snapshot][u] += 1
                in_degrees[t_snapshot][v] += 1
                # if (t >= min_t + (t_snapshot+1)*time_interval):
                #     print ('Snapshot: {}, |V|: {}, |V_s|: {} |E|: {}'.format(t_snapshot, np.max(graphs[-1].edges), len(np.unique(graphs[-1].edges)), len(graphs[-1].edges)))
                #     t_snapshot += 1
                #     graphs.append(new_graph())
                graphs[t_snapshot].add_edge(u, v)

    adjs = [nx.to_scipy_sparse_matrix(graph) for graph in graphs]
    if create_dyn_feats:
        dyn_feats = np.zeros(shape=(n_snapshots, max_node, 16))
        for t in range(n_snapshots):
            graph = graphs[t]
            for u in range(max_node):
                dyn_feats[t, u, 0] = out_weights[t][u]
                dyn_feats[t, u, 1] = in_weights[t][u]
                dyn_feats[t, u, 2] = out_degrees[t][u]
                dyn_feats[t, u, 3] = in_degrees[t][u]
                dyn_feats[t, u, 4] = sum([out_weights[tau][u] for tau in range(t)]) # out weight so far
                dyn_feats[t, u, 5] = sum([in_weights[tau][u] for tau in range(t)]) # in weight so far
                dyn_feats[t, u, 6] = sum([out_degrees[tau][u] for tau in range(t)]) # out degree so far
                dyn_feats[t, u, 7] = sum([in_degrees[tau][u] for tau in range(t)]) # in degree so far
                dyn_feats[t, u, 8] = np.nan_to_num(np.mean([out_weights[t][v] for v in graph.successors(u)]))  # mean out weight of the successors
                dyn_feats[t, u, 9] = np.nan_to_num(np.mean([in_weights[t][v] for v in graph.successors(u)]))  # mean in weight of the successors
                dyn_feats[t, u, 10] = np.nan_to_num(np.mean([out_weights[t][v] for v in graph.predecessors(u)]))  # mean out weight of the predecessors
                dyn_feats[t, u, 11] = np.nan_to_num(np.mean([in_weights[t][v] for v in graph.predecessors(u)]))  # mean in weight of the predecessors
                dyn_feats[t, u, 12] = np.nan_to_num(np.mean([out_degrees[t][v] for v in graph.successors(u)]))  # mean out degrees of the successors
                dyn_feats[t, u, 13] = np.nan_to_num(np.mean([in_degrees[t][v] for v in graph.successors(u)]))  # mean in degrees of the successors
                dyn_feats[t, u, 14] = np.nan_to_num(np.mean([out_degrees[t][v] for v in graph.predecessors(u)]))  # mean out degrees of the predecessors
                dyn_feats[t, u, 15] = np.nan_to_num(np.mean([in_degrees[t][v] for v in graph.predecessors(u)]))  # mean in degrees of the predecessors
        
        np.save('ethereum_phishing/dyn_feats.npy', dyn_feats)
            
    print('Max Node with Deg > 1 per TS: {}'.format(' '.join([str(max(adj.nonzero()[0])) for adj in adjs])))
    return adjs

def print_graph_info(adjs):
    for t, adj in enumerate(adjs):
        edges = adj.nonzero()
        node_set = set()
        for i in range(len(edges[0])):
            node_set.add(edges[0][i])
            node_set.add(edges[1][i])
        print('Snapshot: {}, |V|: {}, |V_s|: {} |E|: {}'.format(t, max(node_set) if (len(node_set) > 0) else 0, len(node_set), len(edges[0])))

if __name__ == '__main__':
    args = get_args()
    datafile, max_t, min_t, max_node, n_edges = get_data_info (args.data_dir, args.dataset, lb_time=args.lb_time, ub_time=args.ub_time)
    if (args.interval is None):
        time_interval = (max_t - min_t) / args.num_graphs
        n_snapshots = args.num_graphs
    else:
        time_interval = args.interval
        time_interval = pd.to_timedelta(int(time_interval[:-1]), unit=time_interval[-1]).total_seconds()
        n_snapshots = math.ceil((max_t - min_t) / time_interval)
    
    print (min_t, max_t, (max_t-min_t)/n_snapshots, n_edges)
    print('Graph Timespan: {}'.format(dt.datetime.utcfromtimestamp(max_t) - dt.datetime.utcfromtimestamp(min_t)))
    print (n_snapshots)

    # if ((args.analyze_only) and (os.path.exists("{}/{}/graphs_{}.pkl".format(args.data_dir, args.dataset, n_snapshots)))):
    #     adjs = pkl.load (open("{}/{}/graphs_{}.pkl".format(args.data_dir, args.dataset, n_snapshots), 'rb'))
    #     print_graph_info(adjs)
    # el
    if (args.analyze_only):
        adjs = get_discrete_graphs (args.data_dir, datafile, args.dataset, max_node, min_t, n_snapshots, time_interval,
                                    create_dyn_feats=args.dyn_feats, discrete=args.discrete, lb_time=args.lb_time, ub_time=args.ub_time)
        print_graph_info(adjs)
    else:
        adjs = get_discrete_graphs (args.data_dir, datafile, args.dataset, max_node, min_t, n_snapshots, time_interval, 
                                    create_dyn_feats=args.dyn_feats, discrete=args.discrete, lb_time=args.lb_time, ub_time=args.ub_time)
        print_graph_info(adjs)
        pkl.dump (adjs, open("{}/{}/graphs_{}.pkl".format(args.data_dir, args.dataset, n_snapshots), "wb"))
    # adjs, node_pairs = load_edge_list(args.data_dir, args.dataset, args.num_graphs, args.task, args.historical_len, args.target_snapshot, lb_time=args.lb_time, ub_time=args.ub_time)
    # try:
    #     os.makedirs ("{}/{}/{}/".format(args.data_dir, args.dataset, args.task))
    # except:
    #     pass
    # pkl.dump (node_pairs, open("{}/{}/{}/nodePairs_h{}_t{}_tot{}.pkl".format(args.data_dir, args.dataset, args.task, args.historical_len, args.target_snapshot, args.num_graphs), "wb"))


# python3 get_data.py --dataset flickr-growth --lb_time 1171018575 --num_graphs 16


# python3 get_data.py --dataset ethereum_phishing --num_graphs 10 --lb_time 1493473765