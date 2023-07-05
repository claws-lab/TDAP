import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import ast
from itertools import islice
from torch import tensor
import torch
import re

def frange (x, y, z):
    l = [x]
    while (l[-1] < y):
        l.append(np.round(l[-1] + z, 2))
    return l

models = ["dysat", "evgcno", "gclstm"]
datasets = ["radoslaw", "opsahl-ucsocial", "reddit_hyperlinks", "DBLP"]

method = 'pgd'
constraint = "noise"

online = False

nn_graphs = [7, 10, 13, 14, 18, 20] #, 39]
ntargets = [1] #, 10, 20] #1
contexts = [6, 9, 12, 13, 17, 19] #, 38]
target_ts = [6, 9, 12, 13, 17, 19] #, 38]

epsilons = frange(0.01, 1.0, 0.01)
epsilon1s = [100] #[2, 20, 100]
all_results_df = defaultdict(lambda: [])

# os.chdir("old_results/")
if method == 'greedy':
    dirs = ["baselines/greedy/results_{}/{}/multi_targets/{}".format(model, dataset, constraint) for dataset in datasets for model in models]
elif method == 'pgd':
    dirs = ["results_{}/{}/multi_targets/pgd/{}".format(model, dataset, constraint) for dataset in datasets for model in models]
else:
    dirs = ["baselines/{}/results_{}/{}/multi_targets/{}".format(method, model, dataset, constraint) for dataset in datasets for model in models]

for res_dir in dirs: 
    ntg = 1
    if 'baselines' in res_dir:
        dataset = res_dir.split("/")[3]
        model = res_dir.split("/")[2].split("_")[1]
    else:
        dataset = res_dir.split("/")[1]
        model = res_dir.split("/")[0].split("_")[1]
    try:
        files = os.listdir(res_dir)
    except:
        continue
    for filename in files:
        try:
            chars = filename[:-4].split("_")
            if online and chars[0] != "onlineResults":
                continue
            elif not (online) and not chars[0].startswith("results"):
                continue
            results_fname = chars[0]
            chars = chars[1:]
            if ("seq" in chars[0]):
                attk_tg = chars[0] 
                chars = chars[1:]
            else:
                attk_tg = "pool"
            if (method in ['random', 'degree']):
                if ('seed' in filename):
                    chars = chars[:-1] + ['l0'] + [chars[-1]]
                else:
                    chars.append("l0")
            if ('seed' in filename):
                if (len(chars) == 8):
                    sampling, ntg = chars[0], int(chars[1][2:])
                    chars = chars[2:]
                elif (len(chars) == 7):
                    if ('tg' in chars[0]):
                        sampling, ntg = "rd", int(chars[0][2:])
                        chars = chars[1:]
                    else:
                        sampling = chars[0]
                        chars = chars[1:]
                elif "tg" in chars[0]:
                    sampling, ntg = "rd", int(chars[0][2:])
                    chars = chars[1:]
                else:
                    sampling = "rd"
            else:
                if (len(chars) == 7):
                    sampling, ntg = chars[0], int(chars[1][2:])
                    chars = chars[2:]
                elif (len(chars) == 6):
                    if ('tg' in chars[0]):
                        sampling, ntg = "rd", int(chars[0][2:])
                        chars = chars[1:]
                    else:
                        sampling = chars[0]
                        chars = chars[1:]
                else:
                    sampling = "rd"
            if (ntg not in ntargets):
                continue
            context, target_snap = chars[1].split("t")
            try:
                num_graphs, context, target_snap, eps, b_eps1, lambda_1, seed = int(chars[0][1:]), int(context[1:]), int(target_snap), float(chars[2][1:]), int(chars[3][2:]), chars[4][1:], int(chars[5][4:])
            except:
                num_graphs, context, target_snap, eps, b_eps1, lambda_1 = int(chars[0][1:]), int(context[1:]), int(target_snap), float(chars[2][1:]), int(chars[3][2:]), chars[4][1:]
                seed = 123
            try:
                lambda_1 = float(lambda_1)
            except:
                lambda_1 = float(0.0)
            if ((num_graphs not in nn_graphs) or (eps not in epsilons) or (b_eps1 not in epsilon1s) or (context not in contexts) or (target_snap not in target_ts)):
                continue
        except:
            continue
        print (res_dir, filename)
        with open("{}/{}".format(res_dir, filename), "r") as f:
            results = defaultdict(lambda: [])
            n_samples, time_taken = 0, 0
            perb_line, perb_metric = "", None
            for line in f:
                if ("Orig AUCROC" in line) or ("Orig Accuracy" in line):
                    orig_auc = float(line.split(": ")[1][:-1])
                    # print ("Orig AUCROC\t{}".format(orig_auc))
                elif ("AUCROC" in line) or ("Accuracy" in line):
                    perb_metric = float(line.split(": ")[1][:-1])
                    # print ("Perb AUCROC\t{}".format(perb_metric))
                elif ("Total time taken" in line):
                    time_taken = float(line.split(": ")[1][:-1])
                elif ("Perturbation" in line):
                    # continue
                    if constraint == 'noise':
                        if ('Target_id' in line):
                            try:
                                perb_line = line.split("Perturbations: ")[1]
                            except:
                                results["aml"].append(int(line.split("Perturbation size: ")[1]))
                                continue
                        else:
                            perb_line = f.readline()[:-1]
                        while True:
                            try:
                                perb_times, perb_targets, perb_nodes, perb_direcs = eval(re.sub("device='cuda:[0-9]'", "device='cpu'", perb_line))
                                break
                            except:
                                line = f.readline()[:-1]
                                if ("Target_id" in line):
                                    print (perb_line)
                                    print(line)
                                    exit()
                                perb_line += line
                        results["aml"].append(perb_times.shape[0])
                    elif constraint == 'noise_feat':
                        results['aml'].append(float(line.split("Perturbation norm: ")[1]))
                elif ((":" in line) and ("," in line)): # and ('tensor' not in line) and ('cuda' not in line) and ('Probs' not in line)):
                    output = line[:-1]
                    k_in_name = False
                    for result in output.split(", "):
                        try:
                            name, value = result.split(": ")
                            if (name == "K"):
                                k_in_name = True
                        except:
                            continue
                        try:
                            value = int(value) if (name == "Pred") else float(value)
                        except:
                            continue
                        results[name].append(value)
                    if (k_in_name):
                        n_samples += 1
            if (perb_metric is None):
                print ("{} running".format(filename))
                continue
            if (len(results["aml"]) != n_samples) or (n_samples !=  len(results['K'])) or (len(results["dz'-dz"]) != n_samples):
                print (filename, "weird")
                continue
            else:
                print (n_samples)
            all_results_df["Results"] += n_samples * [results_fname]
            all_results_df["Num_graphs"] += n_samples * [num_graphs]
            all_results_df["Model"] += n_samples * [model]
            all_results_df["Dataset"] += n_samples * [dataset]
            all_results_df["Method"] += n_samples * [method]
            all_results_df["Seed"] += n_samples * [seed]
            all_results_df["Sequential"] += n_samples * [attk_tg]
            all_results_df["Epsilon"] += n_samples * [eps]
            all_results_df["Epsilon1"] += n_samples * [b_eps1]
            all_results_df["Sampling"] += n_samples * [sampling]
            all_results_df["Ntargets"] += n_samples * [ntg]
            all_results_df["Context"] += n_samples * [context]
            all_results_df["Target_ts"] += n_samples * [target_snap]
            all_results_df["Lambda"] += n_samples * [lambda_1]
            all_results_df["Perf. Drop"] += n_samples * [orig_auc - perb_metric]
            all_results_df["Perf. Drop %"] += n_samples * [(orig_auc - perb_metric)/orig_auc*100]
            all_results_df["Perb Perf."] += n_samples * [perb_metric]
            all_results_df["Orig Perf."] += n_samples * [orig_auc]
            if "Time taken" in results:
                all_results_df["Time taken"] += n_samples * [np.mean(results["Time taken"])]
            else:
                all_results_df["Time taken"] += n_samples * [time_taken/n_samples]
            all_results_df["aml"] += results["aml"] #n_samples * [budget] # 
            all_results_df["K"] += results["K"]
            all_results_df["E"] += results["E"]
            # all_results_df["dz'"] += results["dz'"]
            all_results_df["dz_frac"] += results["dz'/dz"]
            all_results_df["del_dz"] += results["dz'-dz"]

print ([(k, len(v)) for k, v in all_results_df.items()])

if online:
    constraint = "noise_online"

df = pd.DataFrame(all_results_df)  
if (len(datasets) > 0):
    df.to_csv("av_results/all_{}_{}.csv".format(method, constraint), index=False)
else:
    df.to_csv("av_results/all_{}_{}_{}.csv".format(method, datasets[0], constraint), index=False)

df = df.loc[~np.isnan(df["E"])]
df = df.loc[~np.isinf(df["E"])]
dfg = df.groupby(["Results", "Model", "Dataset", "Ntargets", "Method", "Seed", "Num_graphs", "Context", 
                  "Target_ts", "Sequential", "Sampling", "Epsilon", "Epsilon1"])

dfg_mean = dfg.mean().round(3)
dfg_std = dfg.std().round(3)

for k in ["aml", "K", "E", "dz_frac", "del_dz"]:
    dfg_mean[k] = [str(x) + ", " + str(y) for x, y in zip(dfg_mean[k], dfg_std[k])]
    
if (len(datasets) > 1):
    dfg_mean.to_csv("av_results/stats_{}_{}.csv".format(method, constraint), sep="\t")
    dfg.mean().to_csv("av_results/mean_stats_{}_{}.csv".format(method, constraint))
    dfg.std().to_csv("av_results/std_stats_{}_{}.csv".format(method, constraint))
else:
    dfg_mean.to_csv("av_results/stats_{}_{}_{}.csv".format(method, datasets[0], constraint), sep="\t")
    dfg.mean().to_csv("av_results/mean_stats_{}_{}_{}.csv".format(method, datasets[0], constraint))
    dfg.std().to_csv("av_results/std_stats_{}_{}_{}.csv".format(method, datasets[0], constraint))

