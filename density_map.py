import torch
import torch.nn as nn
import torchvision
import numpy as np
from pdb import set_trace as st
import argparse
import pickle
import os
import os.path as osp
import random
import copy
import matplotlib.pyplot as plt
from benchmark import ImageBenchmark

parser = argparse.ArgumentParser()
parser.add_argument("--similar_mode", default="root", choices=["root", "teacher"])
parser.add_argument("--cmp_mode", default="ddv", choices=["ddv", "ddm"])
parser.add_argument("--target_mode", default="targeted", choices=["targeted", "untargeted", "random"])
parser.add_argument("--profiling_mode", default="hybrid", 
                    # choices=["normal", "hybrid"]
                )
parser.add_argument("--pair_mode", default="all")
parser.add_argument("--seed_batch_size", default=50, type=int)
args = parser.parse_args()
args.dir = osp.join( "relation", f"{args.profiling_mode}_{args.cmp_mode}_{args.target_mode}_{args.similar_mode}")
os.makedirs(args.dir, exist_ok=True)

path = osp.join(args.dir, "relation_score.pkl")
with open(path, "rb") as f:
    model_relation_log = pickle.load(f)
path = osp.join("relation", f"relation_{args.pair_mode}_{args.similar_mode}.pkl")
with open(path, "rb") as f:
    model_relation = pickle.load(f)
path = osp.join(args.dir, "relation_score_quant.pkl")
with open(path, "rb") as f:
    model_relation_quant = pickle.load(f)
    
transfer_sim = {
    "transfer": {
            0.1: [],
            0.5: [],
            1: [],
        },
    "prune": {
        0.2: [],
        0.5: [],
        0.8: [],
        },
    "distill": [],
    "steal": {
        "homo": [],
        "heter": [],
    },
    "quant": [],
}
transfer_dissim = copy.deepcopy(transfer_sim)

for i, scores in model_relation_quant.items():
    
    correct = scores["sim_score"] > max(scores["dis_score"])
    confidence = scores["sim_score"] - scores["dis_score_mean"]
    sim_score = scores["sim_score"]
    
    transfer_sim["quant"].append(sim_score)
    transfer_dissim["quant"].append(scores["dis_score_mean"])

for i, scores in model_relation_log.items():
    relation = model_relation[i]
    components = relation["model"].name().split('-')[:-1]
    last_type = components[-1]
    
    correct = scores["sim_score"] > max(scores["dis_score"])
    confidence = scores["sim_score"] - scores["dis_score_mean"]
    sim_score = scores["sim_score"]
    # print(last_type)
    if "transfer" in last_type:
        ratio = float(last_type.split(',')[-1][:-1])
        transfer_sim["transfer"][ratio].append(sim_score)
        # transfer_dissim["transfer"][ratio] += scores["dis_score"]
        transfer_dissim["transfer"][ratio].append(scores["dis_score_mean"])
    elif "prune" in last_type:
        ratio = float(last_type.split('(')[-1].split(')')[0])
        transfer_sim["prune"][ratio].append(sim_score)
        # transfer_dissim["prune"][ratio] += scores["dis_score"]
        transfer_dissim["prune"][ratio].append(scores["dis_score_mean"])
    elif "distill" in last_type:
        transfer_sim["distill"].append(sim_score)
        # transfer_dissim["distill"] += scores["dis_score"]
        transfer_dissim["distill"].append(scores["dis_score_mean"])
    elif "steal" in last_type:
        s_arch = last_type.split('(')[-1].split(')')[0]
        t_arch = components[0].split('(')[-1].split(')')[0].split(',')[0]
        if s_arch == t_arch:
            print(" homo  ", relation["model"].name(), scores["dis_score_mean"])
            transfer_sim["steal"]["homo"].append(sim_score)
            # transfer_dissim["steal"]["homo"] += scores["dis_score"]
            transfer_dissim["steal"]["homo"].append(scores["dis_score_mean"])
        else:
            print("heter  ", relation["model"].name(), scores["dis_score_mean"])
            transfer_sim["steal"]["heter"].append(sim_score)
            # transfer_dissim["steal"]["heter"] += scores["dis_score"]
            transfer_dissim["steal"]["heter"].append(scores["dis_score_mean"])
    else:
        st()
        
plt.figure(figsize=(8,6))
plt_setting = {
    'color': 'blue',
    'alpha': 0.5,
    'marker': 'o',
}
dis_plt_setting = {
    'color': 'red',
    'alpha': 0.5,
    'marker': 'x',
}

IDX = 1
target_list = transfer_sim['transfer'][0.1]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['transfer'][0.1]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 2
target_list = transfer_sim['transfer'][0.5]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['transfer'][0.5]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 3
target_list = transfer_sim['transfer'][1]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['transfer'][1]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 4
target_list = transfer_sim['prune'][0.2]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['prune'][0.2]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 5
target_list = transfer_sim['prune'][0.5]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['prune'][0.5]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 6
target_list = transfer_sim['prune'][0.8]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['prune'][0.8]
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 7
target_list = transfer_sim['quant']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['quant']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 8
target_list = transfer_sim['distill']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['distill']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 9
target_list = transfer_sim['steal']['homo']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['steal']['homo']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)

IDX = 10
target_list = transfer_sim['steal']['heter']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **plt_setting)
target_list = transfer_dissim['steal']['heter']
l = len(target_list)
plt.scatter([IDX]*l, target_list, **dis_plt_setting)


FONTSIZE=25

plt.ylabel("Similarity", fontsize=FONTSIZE)
x_names = [
    "Transfer-0.1", "Transfer-0.5", "Transfer-1",
    "Prune-0.2", "Prune-0.5", "Prune-0.8",
    "Quant", "Distill",
    "Steal-same", "Steal-diff",
]
x_idx = list(range(1,11))
plt.xticks(x_idx, x_names, rotation=90, fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)

plt.tight_layout()
path = osp.join(args.dir, "similarity_dist.pdf")
plt.savefig(path)