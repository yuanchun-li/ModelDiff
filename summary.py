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
from benchmark import ImageBenchmark

parser = argparse.ArgumentParser()
parser.add_argument("--similar_mode", default="root", choices=["root", "teacher"])
parser.add_argument("--cmp_mode", default="ddv", choices=["ddv", "ddm"])
parser.add_argument("--target_mode", default="targeted", 
                    # choices=["targeted", "untargeted", "random"]
                    )
parser.add_argument("--profiling_mode", default="hybrid", 
                    # choices=["normal", "hybrid"]
                )
parser.add_argument("--seed_batch_size", default=50, type=int)
args = parser.parse_args()
args.dir = osp.join( "relation", f"{args.profiling_mode}_{args.cmp_mode}_{args.target_mode}_{args.similar_mode}")
os.makedirs(args.dir, exist_ok=True)

path = osp.join(args.dir, "relation_score.pkl")
with open(path, "rb") as f:
    model_relation_log = pickle.load(f)
path = osp.join("relation", f"relation_{args.similar_mode}.pkl")
with open(path, "rb") as f:
    model_relation = pickle.load(f)
    
transfer_correct = {
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
transfer_confidence = copy.deepcopy(transfer_correct)

for i, scores in model_relation_log.items():
    relation = model_relation[i]
    components = relation["model"].name().split('-')[:-1]
    last_type = components[-1]
    
    correct = scores["sim_score"] > max(scores["dis_score"])
    confidence = scores["sim_score"] - scores["dis_score_mean"]
    # print(last_type)
    if "transfer" in last_type:
        ratio = float(last_type.split(',')[-1][:-1])
        transfer_correct["transfer"][ratio].append(correct)
        transfer_confidence["transfer"][ratio].append(confidence)
    elif "prune" in last_type:
        ratio = float(last_type.split('(')[-1].split(')')[0])
        transfer_correct["prune"][ratio].append(correct)
        transfer_confidence["prune"][ratio].append(confidence)
    elif "distill" in last_type:
        transfer_correct["distill"].append(correct)
        transfer_confidence["distill"].append(confidence)
    elif "steal" in last_type:
        s_arch = last_type.split('(')[-1].split(')')[0]
        t_arch = components[0].split('(')[-1].split(')')[0].split(',')[0]
        if s_arch == t_arch:
            transfer_correct["steal"]["homo"].append(correct)
            transfer_confidence["steal"]["homo"].append(confidence)
        else:
            transfer_correct["steal"]["heter"].append(correct)
            transfer_confidence["steal"]["heter"].append(confidence)
    else:
        st()

# for k in transfer_correct.keys():
#     if isinstance(transfer_correct[k], list):
#         transfer_correct[k] = np.mean(transfer_correct[k])
#     else:
#         for subk in transfer_correct[k].keys():
#             transfer_correct[k][subk] = np.mean(transfer_correct[k][subk])
# for k in transfer_confidence.keys():
#     if isinstance(transfer_confidence[k], list):
#         transfer_confidence[k] = np.mean(transfer_confidence[k])
#     else:
#         for subk in transfer_confidence[k].keys():
#             transfer_confidence[k][subk] = np.mean(transfer_confidence[k][subk])

path = osp.join(args.dir, "summary.txt")
with open(path, "w") as f:
    f.write("Transfer\n")
    
    f.write(f"\t0.1: correct {np.mean(transfer_correct['transfer'][0.1]):.2f}, confidence {np.mean(transfer_confidence['transfer'][0.1]):.2f}\n")
    f.write(f"\t0.5: correct {np.mean(transfer_correct['transfer'][0.5]):.2f}, confidence {np.mean(transfer_confidence['transfer'][0.5]):.2f}\n")
    f.write(f"\t 1 : correct {np.mean(transfer_correct['transfer'][1]):.2f}, confidence {np.mean(transfer_confidence['transfer'][1]):.2f}\n")
    correct_mean = np.mean(transfer_correct['transfer'][0.1] + transfer_correct['transfer'][0.5] + transfer_correct['transfer'][1])
    confidence_mean = np.mean(transfer_confidence['transfer'][0.1] + transfer_confidence['transfer'][0.5] + transfer_confidence['transfer'][1])
    f.write(f"\tall: correct {correct_mean:.2f}, confidence {confidence_mean:.2f}\n")
    
    f.write("Prune\n")
    f.write(f"\t0.2: correct {np.mean(transfer_correct['prune'][0.2]):.2f}, confidence {np.mean(transfer_confidence['prune'][0.2]):.2f}\n")
    f.write(f"\t0.5: correct {np.mean(transfer_correct['prune'][0.5]):.2f}, confidence {np.mean(transfer_confidence['prune'][0.5]):.2f}\n")
    f.write(f"\t0.8: correct {np.mean(transfer_correct['prune'][0.8]):.2f}, confidence {np.mean(transfer_confidence['prune'][0.8]):.2f}\n")
    correct_mean = np.mean(transfer_correct['prune'][0.2] + transfer_correct['prune'][0.5] + transfer_correct['prune'][0.8])
    confidence_mean = np.mean(transfer_confidence['prune'][0.2] + transfer_confidence['prune'][0.5] + transfer_confidence['prune'][0.8])
    f.write(f"\tall: correct {correct_mean:.2f}, confidence {confidence_mean:.2f}\n")
    
    f.write("Distill\n")
    f.write(f"\t correct {np.mean(transfer_correct['distill']):.2f}, confidence {np.mean(transfer_confidence['distill']):.2f}\n")
    f.write("Steal\n")
    f.write(f"\thomo: correct {np.mean(transfer_correct['steal']['homo']):.2f}, confidence {np.mean(transfer_confidence['steal']['homo']):.2f}\n")
    f.write(f"\theter: correct {np.mean(transfer_correct['steal']['heter']):.2f}, confidence {np.mean(transfer_confidence['steal']['heter']):.2f}\n")
    correct_mean = np.mean(transfer_correct['steal']['homo'] + transfer_correct['steal']['heter'])
    confidence_mean = np.mean(transfer_confidence['steal']['homo'] + transfer_confidence['steal']['heter'])
    f.write(f"\tall: correct {correct_mean:.2f}, confidence {confidence_mean:.2f}\n")