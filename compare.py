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

np.random.seed(3)
random.seed(3)
debug=False

name_to_model = {}
bench = ImageBenchmark()
models = list(bench.list_models(fc=False))
for i, model in enumerate(models):
    if not model.torch_model_exists():
        continue
    name_to_model[model.name()] = model
    # print(f'{i}\t {model.__str__()}')
# print(models[1].torch_model)

from modeldiff import ModelDiff
DEVICE = 'cuda'
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

class MultiModel(nn.Module):
    def __init__(self, model1, model2):
        super(MultiModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        
        if out1.shape[1] < out2.shape[1]:
            out = out1 + out2[:,:out1.shape[1]]
        else:
            out = out2 + out1[:,:out2.shape[1]]
        
        
        return out

def compare_with_seed(model1, model2, truth=-1):
    print(f'comparing:\n  model1={model1}\n  model2={model2}')
    md = ModelDiff(model1, model2)
    seed_inputs = md.get_seed_inputs(rand=False)
    sim = md.compute_similarity_with_inputs(seed_inputs)
    if truth == -1:
        truth = 1 if model1.__str__().split('-')[0] == model2.__str__().split('-')[0] else 0
    print(f' similarity is {sim}, truth is {truth}')

def log_scores(relation, relation_log):
    path = osp.join( args.dir, f"relation_score.txt")
    f = open(path, "w")
    
    for i, items in model_relation.items():
        name = items["model"].name()
        similar = items["similar"]
        dissimilar = items["dissimilar"]
        sim_scores = relation_log[i]["sim_score"]
        dis_scores = relation_log[i]["dis_score"]
        dis_score_mean = relation_log[i]["dis_score_mean"]
        f.write(name+"\n")
        f.write("\tsimilar:\n")
        for similar_model, sim_score in zip(similar, sim_scores):
            f.write(f"\t\t{similar_model.name()}, score {sim_score:.3f}\n")
        f.write("\tdissimilar:\n")
        for dissimilar_model, dis_score in zip(dissimilar, dis_scores):
            f.write(f"\t\t{dissimilar_model.name()}, score {dis_score:.3f}\n")
        f.write(f"\t\tdissimilar score mean {dis_score_mean:.3f}\n")
        if debug:
            break
    f.close()
        
    
    
    
def gen_adv_inputs(model, inputs):
    from advertorch.attacks import LinfPGDAttack
    def myloss(yhat, y):
        return -((yhat[:,0]-y[:,0])**2 + 0.1*((yhat[:,1:]-y[:,1:])**2).mean(1)).mean()
        
    model = model.to(DEVICE)
    inputs = torch.from_numpy(inputs).to(DEVICE)
    with torch.no_grad():
        model.eval()
        clean_outputs = model(inputs)
        _, pred = clean_outputs.max(dim=1)
    
    output_shape = clean_outputs.shape
    batch_size = output_shape[0]
    num_classes = output_shape[1]
    
    y = torch.zeros(size=output_shape).to(DEVICE)
    y[:, 0] = 1000
    # y = torch.zeros(size=(batch_size,)).long().to(DEVICE)
    
    # more diversity
#     rand_idx = torch.randint(low=0, high=num_classes, size=(batch_size,))
#     y = torch.nn.functional.one_hot(rand_idx, num_classes=num_classes).to(DEVICE) * 10
#     print(y)
    
    if args.target_mode == "targeted":
        adversary = LinfPGDAttack(
            model, loss_fn=myloss, eps=0.1,
            nb_iter=40, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=True
        )
        adv_inputs = adversary.perturb(inputs, y)
    elif args.target_mode == "untargeted":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1,
            nb_iter=40, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=False
        )
        adv_inputs = adversary.perturb(inputs, pred)
    elif args.target_mode == "random":
        y = torch.zeros(size=output_shape).to(DEVICE)
        idx = torch.randint(low=0, high=output_shape[1], size=output_shape[0:1])
        for i in range(output_shape[0]):
            y[i,idx[i]] = 1000
        adversary = LinfPGDAttack(
            model, loss_fn=myloss, eps=0.1,
            nb_iter=40, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=True
        )
        adv_inputs = adversary.perturb(inputs, y)
    elif args.target_mode == "targeted_nobound":
        adversary = LinfPGDAttack(
            model, loss_fn=myloss, eps=10.0,
            nb_iter=40, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=True
        )
        adv_inputs = adversary.perturb(inputs, y)
    elif args.target_mode == "allone":
        y = torch.zeros(size=output_shape).to(DEVICE) + 1000
        adversary = LinfPGDAttack(
            model, loss_fn=myloss, eps=0.1,
            nb_iter=40, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=True
        )
        adv_inputs = adversary.perturb(inputs, y)
    
    with torch.no_grad():
        model.eval()
        adv_outputs = model(adv_inputs)
        _, adv_pred = adv_outputs.max(dim=1)
        correct = int((adv_pred == pred).sum())
        correct_ratio = correct / adv_pred.shape[0]
        # print(correct_ratio)
    torch.cuda.empty_cache()
    return adv_inputs.to('cpu').numpy()


def compare_with_adv(model1, model2, truth=-1):
    if truth == -1:
        truth = 1 if model1.__str__().split('-')[0] == model2.__str__().split('-')[0] else 0
    print(f'comparing:\n  model1={model1}\n  model2={model2}')
    md = ModelDiff(model1, model2)
    rand = False
    seed_inputs1 = model1.get_seed_inputs(args.seed_batch_size, rand=rand)
    seed_inputs2 = model2.get_seed_inputs(args.seed_batch_size, rand=rand)
    seed_inputs = np.concatenate([seed_inputs1, seed_inputs2])
    
    
    # adv_inputs2 = gen_adv_inputs(model2.torch_model, seed_inputs2)
    # adv_inputs = np.concatenate([adv_inputs1, adv_inputs2])
    
    
    
    if args.profiling_mode == "hybrid":
        adv_inputs1 = gen_adv_inputs(model1.torch_model, seed_inputs1)
        hybrid_inputs = np.concatenate([seed_inputs1, adv_inputs1])
        inputs = hybrid_inputs
    elif args.profiling_mode == "normal":
        inputs = seed_inputs
    elif args.profiling_mode == "half_normal":
        left1 = seed_inputs1.copy()
        left1[:,:,:,112:] = 0
        right1 = seed_inputs1.copy()
        right1[:,:,:,:112] = 0
        
        left2 = seed_inputs2.copy()
        left2[:,:,:,112:] = 0
        right2 = seed_inputs2.copy()
        right2[:,:,:,:112] = 0
        inputs = np.concatenate(
            [left1, left2, right1, right2]
        )
    elif args.profiling_mode == "noise":
        inputs = torch.rand(seed_inputs1.shape)
        inputs = (inputs-0.5) * 2
        inputs *= 2.2
    elif args.profiling_mode == "one_input":
        same_inputs = seed_inputs1
        for i in range(1, args.seed_batch_size):
            same_inputs[i] = same_inputs[0]
        adv_inputs1 = gen_adv_inputs(model1.torch_model, same_inputs)
        hybrid_inputs = np.concatenate([same_inputs, adv_inputs1])
        inputs = hybrid_inputs
    elif args.profiling_mode == "multi_model":
        noise = torch.rand(seed_inputs1.shape).numpy()
        noise = (noise-0.5) * 2
        noise *= 2.2
        multimodel = MultiModel(model1.torch_model, model2.torch_model)
        adv_inputs1 = gen_adv_inputs(multimodel, noise)
        hybrid_inputs = np.concatenate([seed_inputs1, adv_inputs1])
        inputs = hybrid_inputs
    elif args.profiling_mode == "other_domain":
        dataloader = bench.get_dataloader("MIT67", "test", batch_size=args.seed_batch_size)
        images, labels = next(iter(dataloader))
        images = images.to('cpu').numpy()
        
        adv_inputs1 = gen_adv_inputs(model1.torch_model, images)
        hybrid_inputs = np.concatenate([images, adv_inputs1])
        inputs = hybrid_inputs
        



    
    if args.cmp_mode == "ddv":
        sim = md.compute_similarity_with_ddv(inputs)
    elif args.cmp_mode == "ddm":
        sim = md.compute_similarity_with_ddm(inputs)
    # sim = md.compute_similarity_with_ddv(adv_inputs)
    # print(f' compute_similarity_with_ddv,adv_inputs is {sim}, truth is {truth}')
    # sim = md.compute_similarity_with_ddv(seed_inputs)
    # print(f' compute_similarity_with_ddv,seed_inputs is {sim}, truth is {truth}')
    # sim = md.compute_similarity_with_ddv(hybrid_inputs)
    # print(f' compute_similarity_with_ddv,hybrid_inputs is {sim}, truth is {truth}')
    torch.cuda.empty_cache()
    
    return sim
    
    
    
path = osp.join("relation", f"relation_{args.similar_mode}.pkl")
with open(path, "rb") as f:
    model_relation = pickle.load(f)

# path = osp.join("relation", f"relation_{args.similar_mode}_score.txt")
# log_file = open(path, "w")
model_relation_log = {}

length = len(model_relation)
keys = model_relation.keys()
    
# for cnt, (i, relation) in enumerate(model_relation.items()):
for cnt, i in enumerate(keys):
    path = osp.join("relation", f"relation_{args.similar_mode}.pkl")
    with open(path, "rb") as f:
        model_relation = pickle.load(f)
    relation = model_relation[i]
    
    print(f"{cnt}/{len(model_relation)}")
    # if i<23:
    #     continue
    model_relation_log[i] = {}
    model = relation["model"]
    similar_model = relation["similar"][0]
    dissimilar_models = relation["dissimilar"]
    similar_score = compare_with_adv(model, similar_model)
    model_relation_log[i]["sim_score"] = [similar_score]
    
    dissimilar_scores = []
    for dissimilar_model in dissimilar_models:
        dissimilar_score = compare_with_adv(model, dissimilar_model)
        dissimilar_scores.append(dissimilar_score)
    model_relation_log[i]["dis_score"] = dissimilar_scores
    model_relation_log[i]["dis_score_mean"] = np.mean(dissimilar_scores)
    
    
    # log_file.write(model.name()+"\n")
    # log_file.write("\tsimilar:\n")
    # log_file.write(f"\t\t{similar_model.name()}, score {similar_score:.3f}\n")
    # log_file.write("\tdissimilar:\n")
    # for dissimilar_model, dis_score in zip(dissimilar_models, dissimilar_scores):
    #     log_file.write(f"\t\t{dissimilar_model.name()}, score {dis_score:.3f}\n")
    # log_file.write(f"\t\tdissimilar score mean {np.mean(dissimilar_scores):.3f}\n")
    
    del relation, model, similar_model, dissimilar_model
    del model_relation
    torch.cuda.empty_cache()
    if debug:
        break

path = osp.join(args.dir , f"relation_score.pkl")
with open(path, "wb") as f:
    pickle.dump(model_relation_log, f)
path = osp.join("relation", f"relation_{args.similar_mode}.pkl")
with open(path, "rb") as f:
    model_relation = pickle.load(f)
# log_file.close()
log_scores(model_relation, model_relation_log)

path = osp.join(args.dir , f"relation_score.pkl")
with open(path, "rb") as f:
    check = pickle.load(f)
