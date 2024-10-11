import os
import datetime as dt
import networkx as nx
import pickle as pkl
import sys
import argparse


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data', help='Dataset directory')
  parser.add_argument('--dataset', type=str, default='radoslaw', help='Dataset to load.')
  parser.add_argument('--task', type=str, default='link_prediction', help='Downstream Task')
  parser.add_argument('--historical_len', type=int, default=2, help='n_s in the paper, or lookback window.')
  parser.add_argument('--num_graphs', type=int, default=7, help='number of timesteps')
  parser.add_argument('--target_snapshot', type=int, default=3, help='the snapshot being targeted for attack.')
  parser.add_argument('--lb_time', type=int)
  parser.add_argument('--ub_time', type=int)
  return parser.parse_args()


def load_edge_list (data_dir, dataset, n_snapshots, task, historical_len, target_ts, lb_time=None, ub_time=None):
    import os
    import datetime as dt
    import sys
    files = os.listdir(data_dir + "/" + dataset)
    for datafile in files:
        if (("out" in datafile) or ("edges" in datafile)):
            min_t, max_t, max_node = float('inf'), 0, 0
            with open ("{}/{}/{}".format(data_dir, dataset, datafile), "r") as f:
                for line in f:
                    if ('%' not in line):
                        u, v, w, t = [int(x) for x in line[:-1].split()]
                        dt_time = dt.datetime.utcfromtimestamp(t)
                        if ((lb_time is not None) and (ub_time is not None) and (dt_time < lb_time) and (dt_time > ub_time)):
                            continue
                        max_node = max(max_node, u)
                        max_node = max(max_node, v)
                        max_t = max(t, max_t)
                        min_t = min(t, min_t)

            print (dt.datetime.utcfromtimestamp(max_t) - dt.datetime.utcfromtimestamp(min_t))
            print (max_node)
            time_interval = (max_t - min_t)/n_snapshots
            graphs = [nx.DiGraph() for _ in range(n_snapshots)]
            for graph in graphs:
                graph.add_nodes_from(list(range(max_node)))
            node_pairs = []
            last_snapshot = -1
            with open ("{}/{}/{}".format(data_dir, dataset, datafile), "r") as f:
                for line in f:
                    if ('%' not in line):
                        u, v, w, t = [int(x) for x in line[:-1].split()]
                        # starts from 1 usually => but in networkx, would start from 0
                        u, v = u - 1, v - 1
                        dt_time = dt.datetime.utcfromtimestamp(t)
                        if ((lb_time is not None) and (ub_time is not None) and (dt_time < lb_time) and (dt_time > ub_time)):
                            continue
                        t_snapshot = int((t - min_t)/time_interval) if (t != max_t) else n_snapshots-1
                        # if (t_snapshot != last_snapshot):
                        #     # only addition
                        #     graphs[t_snapshot] = graphs[last_snapshot]
                        graphs[t_snapshot].add_edge(u, v)
                        # 
                        if (t_snapshot >= target_ts - historical_len) and (t_snapshot < target_ts):
                            node_pairs.append((u, v, t, w))

            adjs = [nx.to_scipy_sparse_matrix(graph) for graph in graphs]
            return adjs, node_pairs
            
if __name__ == '__main__':
    args = get_args()
    adjs, node_pairs = load_edge_list(args.data_dir, args.dataset, args.num_graphs, args.task, args.historical_len, args.target_snapshot, lb_time=args.lb_time, ub_time=args.ub_time)
    pkl.dump (adjs, open("{}/{}/pre_graphs_{}.pkl".format(args.data_dir, args.dataset, args.num_graphs), "wb"))
    pkl.dump (node_pairs, open("{}/{}/{}/nodePairs_h{}_t{}.pkl".format(args.data_dir, args.dataset, args.task, args.historical_len, args.target_snapshot), "wb"))




import argparse
import math
import pickle as pkl
import networkx as nx
import pandas as pd
import datetime as dt

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='radoslaw', help='Dataset to load.')
  parser.add_argument('--task', type=str, default='link_prediction', help='Downstream Task')
  parser.add_argument('--historical_len', type=int, default=2, help='n_s in the paper, or lookback window.')
  parser.add_argument('--num_graphs', type=int, default=7, help='number of timesteps')
  parser.add_argument('--target_snapshot', type=int, default=3, help='the snapshot being targeted for attack.')
  parser.add_argument('--skip_lines', type=int, default=1, help='Number of lines to skip when reading data file.')
  return parser.parse_args()

def read_file(dataset):
  lines = []
  import os
  for fn in os.listdir(dataset):
    if ("out" in fn) or ("edges" in fn):
      data_file = dataset + '/' + fn
  with open(data_file, 'r') as f:
    for line in f:
      if ('%' not in line):
        line_arr = line[:-1].split()
        line_arr[0] = int(line_arr[0])
        line_arr[1] = int(line_arr[1])
        line_arr[2] = int(line_arr[2])
        line_arr[3] = int(line_arr[3])
        lines.append(line_arr)
  df = pd.DataFrame(lines, columns=['source', 'target', 'weight', 'timestamp'])
  return df

def make_graphs(df, num_graphs, target_snapshot, historical_len):
  min_ts = df.timestamp.min()
  max_ts = df.timestamp.max()
  print (dt.datetime.utcfromtimestamp(max_ts) - dt.datetime.utcfromtimestamp(min_ts))
  interval = (max_ts - min_ts) / num_graphs
  max_node = max(df["source"].max(), df["target"].max()) 
  print (max_node)

  graphs = [nx.DiGraph() for _ in range(num_graphs)]
  for graph in graphs:
    graph.add_nodes_from(list(range(max_node)))
  node_pairs = []
  for row in df.iterrows():
    source = row[1]['source'] - 1
    target = row[1]['target'] - 1
    weight = row[1]['weight']
    timestamp = row[1]['timestamp']

    if timestamp == max_ts:
      timestep = num_graphs - 1
    else:
      timestep = int((timestamp - min_ts) // interval)
    
    graphs[timestep].add_edge(source, target, weight=weight)

    if timestep >= target_snapshot - historical_len and timestep < target_snapshot:
      node_pairs.append((source, target, timestamp, weight))
    
  for graph in graphs:
    nodes = list(graph.nodes())
    nodes = [int(node) for node in nodes]
    nodes.sort()
    print(nodes)
  
  adjs = [nx.to_scipy_sparse_matrix(graph) for graph in graphs]
  return adjs, node_pairs

if __name__ == "__main__":
  args = get_args()
  df = read_file(args.dataset)
  graphs, node_pairs = make_graphs(df, args.num_graphs, args.target_snapshot, args.historical_len)
  
  with open('{}/graphs_{}.pkl'.format(args.dataset, args.num_graphs), 'wb') as f:
    pkl.dump(graphs, f)

  with open('{}/{}/nodePairs_{}.pkl'.format(args.dataset, args.task, args.historical_len), 'wb') as f:
    pkl.dump(node_pairs, f)

