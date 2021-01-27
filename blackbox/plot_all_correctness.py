import torch
import torch.nn as nn
import torchvision
import numpy as np
from pdb import set_trace as st
import argparse
import os
import os.path as osp
import pickle
import random
import matplotlib.pyplot as plt

from benchmark import ImageBenchmark

from test_blackbox_compare import expand_vector, evaluate_inputs

np.random.seed(3)
random.seed(3)

parser = argparse.ArgumentParser()
parser.add_argument("--root_model_idx", default=2)
parser.add_argument("--eps", default=1, type=int,)
parser.add_argument("--save_dir", default="results/blackbox_ablation")
parser.add_argument("--max_iter", type=int, default=10000)
parser.add_argument("--log_every", type=int, default=100)
# parser.add_argument("--max_iter", type=int, default=100)
# parser.add_argument("--log_every", type=int, default=10)
args = parser.parse_args()
root_models = ['pretrain(resnet18,ImageNet)-', 'pretrain(mbnetv2,ImageNet)-', 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-']
root_model_name = root_models[args.root_model_idx]

path = osp.join(args.save_dir, "all_correctness", f"{root_model_name}",  "relation.pkl")
with open(path, "rb") as f:
    relations = pickle.load(f)
    
path = osp.join(args.save_dir, "all_correctness", f"{root_model_name}",  "eval.pkl")
with open(path, "rb") as f:
    evaluation = pickle.load(f)
    
sim_mean, dissim_mean = [], []
model_correct = []
for relation in relations:
    all_sim = relation["similar_scores"]
    all_dissim = relation["dissimilar_scores"]
    
    if len(all_sim) == 200:
        all_sim = all_sim[::2]
        all_dissim = all_dissim[::2]
    elif len(all_sim) == 1000:
        all_sim = all_sim[::10]
        all_dissim = all_dissim[::10]

    all_correct = []
    for sim, dissim in zip(all_sim, all_dissim):
        
        teacher_correct = int( sim[0] > max(dissim[:5]) )
        root_correct = int( sim[1] > max(dissim[5:]) )
        # all_correct.append(np.mean([teacher_correct, root_correct]))
        all_correct.append(np.mean([ teacher_correct, root_correct]))
    print(len(all_correct))
    model_correct.append(np.array(all_correct))
    
    sim_mean.append(all_sim)
sim_mean = np.stack(sim_mean)
sim_mean = sim_mean.mean(axis=0).mean(axis=-1)


model_correct = np.stack(model_correct)

correct = np.mean(model_correct, axis=0)
        
    
    
iters, divergence, diversity, succ = [], [], [], [],

for i, eval in evaluation.items():
    iters.append(i)
    divergence.append(float(eval["divergence"]))
    diversity.append(float(eval["diversity"]))
    succ.append(float(eval["succ"].sum())/100)
    
    
FONTSIZE=10
LINEWIDTH=2

gap = [s - d for s, d in zip(sim, dissim)]
plt.plot(iters, correct.tolist(), label="Correctness", alpha=0.5, linewidth=LINEWIDTH)
plt.plot(iters, succ, label="Success", alpha=0.5, linewidth=LINEWIDTH)
plt.plot(iters, divergence, label="Divergence", alpha=0.5, linewidth=LINEWIDTH)
plt.plot(iters, diversity, label="Diversity", alpha=0.5, linewidth=LINEWIDTH)

plt.xlabel("Iteration", fontsize=FONTSIZE)

plt.legend(fontsize=FONTSIZE)

plt.tight_layout()
path = osp.join(args.save_dir, "all_correctness", f"{root_model_name}", "iter.pdf")
plt.savefig(path)
plt.clf()

FONTSIZE=15
LINEWIDTH=4

l = int(len(iters)/2)
score = [gen + 0.5*ver for gen, ver in zip(divergence, diversity)]
plt.plot(iters[:l], correct.tolist()[:l], label="Correctness", alpha=0.5, linewidth=LINEWIDTH)
plt.plot(iters[:l], score[:l], label="Score", alpha=0.5, linewidth=LINEWIDTH, linestyle="--")
plt.plot(iters[:l], sim_mean.tolist()[:l], label="Similarity", alpha=0.5, linewidth=LINEWIDTH, linestyle=":")

plt.xlabel("Iteration", fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)

plt.legend(fontsize=FONTSIZE)

plt.tight_layout()
path = osp.join(args.save_dir, "all_correctness", f"{root_model_name}", "progressive_correctness.pdf")
plt.savefig(path)