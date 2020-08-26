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
}
transfer_dissim = copy.deepcopy(transfer_sim)

transfer_ratio = {
    "SDog120": {},
    "Flower102": {},
}
prune_ratio = {
    "SDog120": {},
    "Flower102": {},
}

for i, scores in model_relation_log.items():
    relation = model_relation[i]
    components = relation["model"].name().split('-')[:-1]
    last_type = components[-1]
    
    correct = scores["sim_score"] > max(scores["dis_score"])
    confidence = scores["sim_score"] - scores["dis_score_mean"]
    sim_score = scores["sim_score"]
    
    if "transfer" in last_type:
        dataset = last_type.split('(')[-1].split(',')[0]
        ratio = float(last_type.split(',')[-1][:-1])
        if ratio in transfer_ratio[dataset]:
            transfer_ratio[dataset][ratio] += sim_score
        else:
            transfer_ratio[dataset][ratio] = sim_score
    elif "prune" in last_type:
        
        ratio = float(last_type.split('(')[-1].split(')')[0])
        dataset = components[-2].split('(')[-1].split(',')[0]
        if ratio in prune_ratio[dataset]:
            prune_ratio[dataset][ratio] += sim_score
        else:
            prune_ratio[dataset][ratio] = sim_score
raw_transfer = copy.deepcopy(transfer_ratio)
raw_prune = copy.deepcopy(prune_ratio)


for k in transfer_ratio.keys():
    for ratio in transfer_ratio[k].keys():
        transfer_ratio[k][ratio] = np.mean(transfer_ratio[k][ratio])
for k in prune_ratio.keys():
    for ratio in prune_ratio[k].keys():
        prune_ratio[k][ratio] = np.mean(prune_ratio[k][ratio])

FONTSIZE=30
LINEWIDTH=6
plt.figure(figsize=(16,6))

plt.subplot(121)
x = transfer_ratio["SDog120"].keys()
y = transfer_ratio["SDog120"].values()
idx = range(len(y))
plt.plot(idx, y, label="Dog", color="red", alpha=0.5, linewidth=LINEWIDTH)


x = transfer_ratio["Flower102"].keys()
y = transfer_ratio["Flower102"].values()
idx = range(len(y))
plt.plot(idx, y, label="Flower", color="green", alpha=0.5, linewidth=LINEWIDTH)

plt.xticks(idx, x, fontsize=FONTSIZE)
plt.xlabel("Transfer ratio", fontsize=FONTSIZE)
plt.yticks([0.94, 0.95, 0.96, 0.97, 0.98], fontsize=FONTSIZE)
plt.ylabel("Similarity", fontsize=FONTSIZE)


plt.subplot(122)
x = prune_ratio["SDog120"].keys()
y = prune_ratio["SDog120"].values()
idx = range(len(y))
plt.plot(idx, y, label="Dog", color="red", alpha=0.5, linewidth=LINEWIDTH)


x = prune_ratio["Flower102"].keys()
y = prune_ratio["Flower102"].values()
idx = range(len(y))
plt.plot(idx, y, label="Flower", color="green", alpha=0.5, linewidth=LINEWIDTH)

plt.xticks(idx, x, fontsize=FONTSIZE)
plt.xlabel("Prune ratio", fontsize=FONTSIZE)
plt.yticks([0.91, 0.93, 0.95, 0.97], fontsize=FONTSIZE)


plt.legend(loc="lower left", fontsize=FONTSIZE)

plt.tight_layout()
path = osp.join(args.dir, "ratio.pdf")
plt.savefig(path)
