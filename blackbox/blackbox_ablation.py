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

from benchmark import ImageBenchmark

from test_blackbox_compare import expand_vector, evaluate_inputs

np.random.seed(3)
random.seed(3)

parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", default='pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-')
# parser.add_argument("--model_name", default='pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-')
parser.add_argument("--model_name", default='pretrain(resnet18,ImageNet)-')
parser.add_argument("--model_idx", default=None)
parser.add_argument("--save_dir", default="results/blackbox_ablation")
parser.add_argument("--eps", default=1, type=int,)
parser.add_argument("--max_iter", type=int, default=100000)
parser.add_argument("--log_every", type=int, default=1000)
# parser.add_argument("--max_iter", type=int, default=100)
# parser.add_argument("--log_every", type=int, default=10)
args = parser.parse_args()
args.save_dir = osp.join(args.save_dir, f"{args.model_name}eps={args.eps}")
os.makedirs(args.save_dir, exist_ok=True)

bench = ImageBenchmark()
models = list(bench.list_models())
models_dict = {}
model_names = []
for i, model in enumerate(models):
    if not model.torch_model_exists():
        continue
#     print(f'{i}\t {model.__str__()}')
    models_dict[model.__str__()] = model
    model_names.append(model.__str__())
# for i, name in enumerate(model_names):
#     if name == args.model_name:
#         print(i, name)
# 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-', 42

from modeldiff import ModelDiff
DEVICE = 'cuda'
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
image_size = 224

def load_model_relation():
    pair_mode = "all"
    similar_mode = "teacher"
    path = osp.join("relation", f"relation_{pair_mode}_{similar_mode}.pkl")
    with open(path, "rb") as f:
        teacher_model_relation = pickle.load(f)
    
    similar_mode = "root"
    path = osp.join("relation", f"relation_{pair_mode}_{similar_mode}.pkl")
    with open(path, "rb") as f:
        root_model_relation = pickle.load(f)
    
    model_relations = {}
    for idx in teacher_model_relation.keys():
        relation = teacher_model_relation[idx]
        relation["similar"].append(root_model_relation[idx]["similar"][0])
        model_relations[idx] = relation
    return model_relations
        
model_relations = load_model_relation()

def compute_similarity(relation, seed_inputs, adv_inputs, evaluation):
    seed_inputs = seed_inputs.cpu().numpy()
    adv_inputs = adv_inputs.cpu().numpy()
    inputs = np.concatenate([seed_inputs, adv_inputs])
    
    model = relation["model"]
    
    with torch.no_grad():
        similar_dist = {}
        for target_model in relation["similar"]:
            md = ModelDiff(model, target_model)
            sim = md.compute_similarity_with_ddv(inputs)
            similar_dist[target_model.name()] = sim
        dissimilar_dist = {}
        for target_model in relation["dissimilar"]:
            md = ModelDiff(model, target_model)
            sim = md.compute_similarity_with_ddv(inputs)
            dissimilar_dist[target_model.name()] = sim
        evaluation["similar"] = similar_dist
        evaluation["dissimilar"] = dissimilar_dist
        evaluation["similar_score"] = np.mean(list(similar_dist.values()))
        evaluation["dissimilar_score"] = np.mean(list(dissimilar_dist.values()))
        # eval_line = f'score={score:.4f}, divergence={divergence:.4f}, diversity={diversity:.4f}, num_succ={succ.sum()}'
        eval_line = (
            evaluation["eval_line"] + 
            f", sim={evaluation['similar_score']:.4f}, dissim={evaluation['dissimilar_score']:.4f}"
        )
        evaluation["eval_line"] = eval_line
        return evaluation

    
def optimize_towards_goal(
    model, seed_inputs, seed_outputs, seed_preds, relation,
    max_iters=args.max_iter, epsilon=1, log_every=args.log_every
):
#     seed_inputs = torch.from_numpy(seed_inputs).to(DEVICE)
#     seed_outputs = torch.from_numpy(seed_outputs).to(DEVICE)
#     seed_preds = torch.from_numpy(seed_preds).to(DEVICE)
    input_shape = seed_inputs[0].shape
    n_inputs = seed_inputs.shape[0]
    ndims = np.prod(input_shape)
    eval_log = {}
    last_eval = None
    
    with torch.no_grad():
        inputs = seed_inputs.clone()
        evaluation = evaluate_inputs(model, inputs, seed_outputs, seed_preds)
        print(f'initial_evaluation: {evaluation["eval_line"]}')
        
        for i in range(max_iters):
#             print(f'mutation {i}-th iteration')

            mutation_pos = np.random.randint(0, ndims)
            mutation = np.zeros(ndims).astype(np.float32)
            
            mutation[mutation_pos] = epsilon
            mutation = np.reshape(mutation, input_shape)

            mutation_batch = np.zeros(shape=inputs.shape).astype(np.float32)
#             all_indices = list(range(0, n_inputs))
#             mutation_indices = np.random.choice(all_indices, size=int(n_inputs * 0.85), replace=False)
#             print(mutation_indices)
#             mutation_idx = np.random.randint(0, n_inputs)
            
            mutation_indices = evaluation['remaining']
            mutation_batch[mutation_indices] = mutation
            mutation_batch = torch.from_numpy(mutation_batch).to(DEVICE)

            prev_score = evaluation["score"]
            mutate_right_inputs = (inputs + mutation_batch).clamp(inputs.min(), inputs.max())
            mutate_right_eval = evaluate_inputs(model, mutate_right_inputs, seed_outputs, seed_preds)
            mutate_right_score = mutate_right_eval['score']
            mutate_left_inputs = (inputs - mutation_batch).clamp(inputs.min(), inputs.max())
            mutate_left_eval = evaluate_inputs(model, mutate_left_inputs, seed_outputs, seed_preds)
            mutate_left_score = mutate_left_eval['score']

            if mutate_right_score <= prev_score and mutate_left_score <= prev_score:
                evaluation = last_eval
                # continue
            if mutate_right_score > mutate_left_score:
#                 print(f'mutate right: {prev_score}->{mutate_right_score}')
                inputs = mutate_right_inputs
                evaluation = mutate_right_eval
            else:
#                 print(f'mutate left: {prev_score}->{mutate_left_score}')
                inputs = mutate_left_inputs
                evaluation = mutate_left_eval
            if (i+1) % log_every == 0:
                # noise = torch.rand(seed_inputs.shape).to(DEVICE)
                # noise = (noise-0.5) * 2
                # noise = noise + seed_inputs
                
                # evaluation = compute_similarity(
                #     relation, seed_inputs, inputs, evaluation,
                # )
                print(f'{i:4d}-th evaluation: {evaluation["eval_line"]}')
                eval_log[i] = evaluation
                last_eval = evaluation
        return inputs, eval_log


def model_ablation():
    model = models_dict[args.model_name]
    
    relation = None
    for idx in model_relations.keys():
        if model_relations[idx]["model"].name() == model.name():
            relation = model_relations[idx]
            break
    # if relation is None:
    #     raise RuntimeError("Model relation not exist")
    
    
    seed_inputs = model.get_seed_inputs(100, rand=False)
    seed_inputs = torch.from_numpy(seed_inputs).to(DEVICE)
    seed_outputs = model.batch_forward(seed_inputs)
    _, seed_preds = seed_outputs.data.max(1)
    
    adv_inputs, eval_log = optimize_towards_goal(
        model.torch_model_on_device, seed_inputs, seed_outputs, seed_preds, relation
    )
    adv_outputs = model.batch_forward(adv_inputs).cpu()
    _, adv_preds = adv_outputs.data.max(1)

    print(f"seed_preds={seed_preds}, adv_preds={adv_preds}")
    
    path = osp.join(args.save_dir, f"eval.pkl")
    with open(path, "wb") as f:
        pickle.dump(eval_log, f)
    
if __name__=="__main__":
    model_ablation()