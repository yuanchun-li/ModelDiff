import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
from pdb import set_trace as st
import pickle

from benchmark import ImageBenchmark

np.random.seed(3)
random.seed(3)

from valid import *

def model_dissimilar(model1, model2):
    model1_name = model1.name()
    model2_name = model2.name()
    
    if model1_name.split('-')[0] != model2_name.split('-')[0]:
        return True
    else:
        return False

def print_relation(model_relation):
    for i, items in model_relation.items():
        name = items["model"].name()
        similar = items["similar"]
        dissimilar = items["dissimilar"]
        print(name)
        print("\tsimilar:")
        for similar_model in similar:
            print(f"\t\t{similar_model.name()}")
        print("\tdissimilar:")
        for dissimilar_model in dissimilar:
            print(f"\t\t{dissimilar_model.name()}")

name_to_model = {}
bench = ImageBenchmark()
models = list(bench.list_models(fc=False))
for i, model in enumerate(models):
    if not valid_model(model):
        continue
    name_to_model[model.name()] = model
    # print(model.name())
    # print(f'{i}\t {model}')
# print(models[1].torch_model)

dissim_arch_list = {
    "mbnetv2": [],
    "resnet18": [],
}
for model in models:
    # if "pretrain" in model.name():
    #     continue
    name = model.name()
    components = model.name().split('-')[:-1]
    if "steal" in name:
        arch = components[-1]
    else:
        arch = components[0]

    if "resnet18" in arch:
        dissim_arch_list["resnet18"].append(model)
    elif "mbnetv2" in arch:
        dissim_arch_list["mbnetv2"].append(model)
    else:
        raise RuntimeError


model_relation = {}
for i, model in enumerate(models):
    if not valid_quant_model(model):
        continue
    model_components = model.name().split('-')
    if len(model_components) == 2:
        continue
    if "steal" in model.name():
        arch = model_components[-2].split(')')[0].split('(')[1]
    else:
        arch = model_components[0].split(',')[0].split('(')[1]
    base_model_name = model_components[0]+'-'
    if arch not in base_model_name:
        continue
            
    base_model = name_to_model[base_model_name]
    dataset = model_components[1].split(',')[0].split('(')[1]
    
        

    model_relation[i] = {
        "model": model,
        "similar": [base_model],
        "dissimilar": [],
    }
    
    while True:
        pair_model = random.choice(dissim_arch_list[arch])
        # print(pair_model.name())
        # print(not valid_model(pair_model), (pair_model.name() == model.name()), (dataset not in pair_model.name()) )
        if (not valid_model(pair_model)) or (pair_model.name() == model.name()) or pair_model in model_relation[i]["dissimilar"] :
            continue
        pair_components = pair_model.name().split('-')
        if model_components[0] != pair_components[0]:
            if len(model_relation[i]["dissimilar"]) < 5:
                model_relation[i]["dissimilar"].append(pair_model)
        if len(model_relation[i]["dissimilar"]) >= 5:
            break
    # break

with open(f"relation/relation_arch_root.pkl", "wb") as f:
    pickle.dump(model_relation, f)
print_relation(model_relation)

