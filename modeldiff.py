#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import time
import logging
import pathlib
import tempfile
import copy
import random
import torch
import numpy as np
import tensorflow as tf
from scipy import spatial
from abc import ABC, abstractmethod

from utils import lazy_property, Utils


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelComparison(ABC):
    def __init__(self, model1, model2):
        self.logger = logging.getLogger('ModelComparison')
        self.model1 = model1
        self.model2 = model2

    @abstractmethod
    def compare(self):
        pass


class ModelDiff(ModelComparison):
    N_INPUT_PAIRS = 100
    MAX_VAL = 256

    def __init__(self, model1, model2, gen_inputs=None, input_metrics=None, compute_decision_dist=None, compare_ddv=None):
        super().__init__(model1, model2)
        self.logger = logging.getLogger('ModelDiff')
        self.logger.info(f'comparing {model1} and {model2}')
        self.logger.debug(f'initialize comparison: {self.model1} {self.model2}')
        self.logger.debug(f'input shapes: {self.model1.input_shape} {self.model2.input_shape}')
        self.input_shape = model1.input_shape
        if list(model1.input_shape) != list(model2.input_shape):
            self.logger.warning('input shapes do not match')
        self.gen_inputs = gen_inputs if gen_inputs else ModelDiff._gen_profiling_inputs_search
        self.input_metrics = input_metrics if input_metrics else ModelDiff.metrics_output_diversity
        self.compute_decision_dist = compute_decision_dist if compute_decision_dist else ModelDiff._compute_decision_dist_output_cos
        self.compare_ddv = compare_ddv if compare_ddv else ModelDiff._compare_ddv_cos

    def compare(self, use_torch=True):
        self.logger.info(f'generating seed inputs')
        rand = False
        seed_inputs = np.concatenate([
            self.model1.get_seed_inputs(self.N_INPUT_PAIRS, rand=rand),
            self.model2.get_seed_inputs(self.N_INPUT_PAIRS, rand=rand)
        ])
        seed_inputs = list(seed_inputs)
        np.random.shuffle(seed_inputs)
        seed_inputs = np.array(seed_inputs)
        if use_torch:
            seed_inputs = torch.from_numpy(seed_inputs)
        self.logger.info(f'  seed inputs generated with shape {seed_inputs.shape}')

        self.logger.info(f'generating profiling inputs')
        profiling_inputs = self.gen_inputs(self, seed_inputs, use_torch=use_torch)
        # input_pairs = []
        # for i in range(int(len(profiling_inputs) / 2)):
        #     xa = profiling_inputs[2 * i]
        #     xb = profiling_inputs[2 * i + 1]
        #     xa = np.expand_dims(xa, axis=0)
        #     xb = np.expand_dims(xb, axis=0)
        #     input_pairs.append((xa, xb))
        self.logger.info(f'  profiling inputs generated with shape {profiling_inputs.shape}')

        self.logger.info(f'computing metrics')
        input_metrics_1 = self.input_metrics(self.model1, profiling_inputs, use_torch=use_torch)
        input_metrics_2 = self.input_metrics(self.model2, profiling_inputs, use_torch=use_torch)
        self.logger.info(f'  input metrics: model1={input_metrics_1} model2={input_metrics_2}')

        model_similarity = self._compute_distance(profiling_inputs)
        return model_similarity
    
    def _compute_distance(self, profiling_inputs):
        self.logger.info(f'computing DDVs')
        ddv1 = []  # DDV is short for decision distance vector
        ddv2 = []
        profiling_outputs_1 = self.model1.batch_forward(profiling_inputs)
        profiling_outputs_2 = self.model2.batch_forward(profiling_inputs)
        self.logger.debug(
            f'{self.model1}: \n profiling_outputs_1={profiling_outputs_1.shape}\n{profiling_outputs_1}\n'
            f'{self.model2}: \n profiling_outputs_2={profiling_outputs_2.shape}\n{profiling_outputs_2}'
        )
        profiling_outputs_1 = profiling_outputs_1.to('cpu').numpy()
        profiling_outputs_2 = profiling_outputs_2.to('cpu').numpy()
        for i in range(int(len(profiling_inputs) / 2)):
            # self.logger.info(f'generated input pair:\n{xa}\n{xb}')
            y1a = profiling_outputs_1[2 * i]
            y1b = profiling_outputs_1[2 * i + 1]
            dist1 = spatial.distance.euclidean(y1a, y1b)

            y2a = profiling_outputs_2[2 * i]
            y2b = profiling_outputs_2[2 * i + 1]
            dist2 = spatial.distance.euclidean(y2a, y2b)
            # dist1 = self.compute_decision_dist(self.model1, xa, xb)
            # dist2 = self.compute_decision_dist(self.model2, xa, xb)
            # self.logger.debug(f'computed distances: {dist1} {dist2}')
            ddv1.append(dist1)
            ddv2.append(dist2)
        ddv1, ddv2 = np.array(ddv1), np.array(ddv2)
        self.logger.info(f'  DDV computed: shape={ddv1.shape} and {ddv2.shape}')

        self.logger.info(f'measuring model similarity')
        ddv1 = Utils.normalize(np.array(ddv1))
        ddv2 = Utils.normalize(np.array(ddv2))
        self.logger.debug(f'ddv1={ddv1}\nddv2={ddv2}')
        ddv_distance = self.compare_ddv(ddv1, ddv2)
        model_similarity = 1 - ddv_distance
        self.logger.info(f'  model similarity: {model_similarity}')
        return model_similarity

    @staticmethod
    def metrics_output_diversity(model, inputs, use_torch=False):
        outputs = model.batch_forward(inputs).to('cpu').numpy()
#         output_dists = []
#         for i in range(0, len(outputs) - 1):
#             for j in range(i + 1, len(outputs)):
#                 output_dist = spatial.distance.euclidean(outputs[i], outputs[j])
#                 output_dists.append(output_dist)
#         diversity = sum(output_dists) / len(output_dists)
        output_dists = spatial.distance.cdist(list(outputs), list(outputs), p=2.0)
        diversity = np.mean(output_dists)
        return diversity

    @staticmethod
    def metrics_output_variance(model, inputs, use_torch=False):
        batch_output = model.batch_forward(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        # print(batch_output_mean.shape)
        output_variances = np.var(batch_output_mean, axis=0)
        # print(output_variances)
        return np.mean(output_variances)

    @staticmethod
    def metrics_output_range(model, inputs, use_torch=False):
        batch_output = model.batch_forward(inputs).to('cpu').numpy()
        mean_axis = tuple(list(range(len(batch_output.shape)))[2:])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        output_ranges = np.max(batch_output_mean, axis=0) - np.min(batch_output_mean, axis=0)
        return np.mean(output_ranges)

    @staticmethod
    def metrics_neuron_coverage(model, inputs, use_torch=False):
        module_irs = model.batch_forward_with_ir(inputs)
        neurons = []
        neurons_covered = []
        for module in module_irs:
            ir = module_irs[module]
            # print(f'{tensor["name"]} {batch_tensor_value.shape}')
            # if 'relu' not in tensor["name"].lower():
            #     continue
            squeeze_axis = tuple(list(range(len(ir.shape)))[:-1])
            squeeze_ir = np.max(ir, axis=squeeze_axis)
            for i in range(squeeze_ir.shape[-1]):
                neuron_name = f'{module}-{i}'
                neurons.append(neuron_name)
                neuron_value = squeeze_ir[i]
                covered = neuron_value > 0.1
                if covered:
                    neurons_covered.append(neuron_name)
        neurons_not_covered = [neuron for neuron in neurons if neuron not in neurons_covered]
        print(f'{len(neurons_not_covered)} neurons not covered: {neurons_not_covered}')
        return float(len(neurons_covered)) / len(neurons)

    @staticmethod
    def _compute_decision_dist_output_cos(model, xa, xb):
        ya = model.batch_forward(xa)
        yb = model.batch_forward(xb)
        return spatial.distance.cosine(ya, yb)

    @staticmethod
    def _gen_profiling_inputs_none(comparator, seed_inputs, use_torch=False):
        return seed_inputs

    @staticmethod
    def _gen_profiling_inputs_random(comparator, seed_inputs, use_torch=False):
        return np.random.normal(size=seed_inputs.shape).astype(np.float32)

    # @staticmethod
    # def _gen_profiling_inputs_1pixel(comparator, seed_inputs):
    #     input_shape = seed_inputs[0].shape
    #     for i in range(len(seed_inputs)):
    #         x = np.zeros(input_shape, dtype=np.float32)
    #         random_index = np.unravel_index(np.argmax(np.random.normal(size=input_shape)), input_shape)
    #         x[random_index] = 1
    #         yield x

    @staticmethod
    def _gen_profiling_inputs_search(comparator, seed_inputs, use_torch=False, epsilon=0.2):
        input_shape = seed_inputs[0].shape
        max_iterations = 1000
        max_steps = 10
        model1 = comparator.model1
        model2 = comparator.model2
        
        ndims = np.prod(input_shape)
#         mutate_positions = torch.randperm(ndims)

        initial_outputs1 = model1.batch_forward(seed_inputs).to('cpu').numpy()
        initial_outputs2 = model2.batch_forward(seed_inputs).to('cpu').numpy()
        
        def evaluate_inputs(inputs):
            outputs1 = model1.batch_forward(inputs).to('cpu').numpy()
            outputs2 = model2.batch_forward(inputs).to('cpu').numpy()
            metrics1 = comparator.input_metrics(comparator.model1, inputs)
            metrics2 = comparator.input_metrics(comparator.model2, inputs)

            output_dist1 = np.mean(spatial.distance.cdist(
                list(outputs1),
                list(initial_outputs1),
                p=2).diagonal())
            output_dist2 = np.mean(spatial.distance.cdist(
                list(outputs2),
                list(initial_outputs2),
                p=2).diagonal())
            print(f'  output distance: {output_dist1},{output_dist2}')
            print(f'  metrics: {metrics1},{metrics2}')
            # if mutated_metrics <= metrics:
            #     break
            return output_dist1 * output_dist2 * metrics1 * metrics2
        
        inputs = seed_inputs
        score = evaluate_inputs(inputs)
        print(f'score={score}')
        
        for i in range(max_iterations):
            comparator._compute_distance(inputs)
            print(f'mutation {i}-th iteration')
            # mutation_idx = random.randint(0, len(inputs))
            # mutation = np.random.random_sample(size=input_shape).astype(np.float32)
            
            mutation_pos = np.random.randint(0, ndims)
            mutation = np.zeros(ndims).astype(np.float32)
            mutation[mutation_pos] = epsilon
            mutation = np.reshape(mutation, input_shape)
            
            # print(f'{inputs.shape} {mutation_perturbation.shape}')
            # for j in range(max_steps):
                # mutated_inputs = np.clip(inputs + mutation, 0, 1)
                # print(f'{list(inputs)[0].shape}')
            mutate_right_inputs = inputs + mutation
            mutate_right_score = evaluate_inputs(mutate_right_inputs)
            mutate_left_inputs = inputs - mutation
            mutate_left_score = evaluate_inputs(mutate_left_inputs)
            
            if mutate_right_score <= score and mutate_left_score <= score:
                continue
            if mutate_right_score > mutate_left_score:
                print(f'mutate right: {score}->{mutate_right_score}')
                inputs = mutate_right_inputs
                score = mutate_right_score
            else:
                print(f'mutate left: {score}->{mutate_left_score}')
                inputs = mutate_left_inputs
                score = mutate_left_score
        return inputs
        
    @staticmethod
    def _compare_ddv_cos(ddv1, ddv2):
        return spatial.distance.cosine(ddv1, ddv2)


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Compare similarity between two models.")

    parser.add_argument("-benchmark_dir", action="store", dest="benchmark_dir",
                        required=False, default=".", help="Path to the benchmark.")
    parser.add_argument("-model1", action="store", dest="model1",
                        required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2",
                        required=True, help="model 2.")
    args, unknown = parser.parse_known_args()
    return args


def evaluate_micro_benchmark():
    lines = pathlib.Path('benchmark_models/model_pairs.txt').read_text().splitlines()
    eval_lines = []
    for line in lines:
        model1_str = line.split()[0]
        model2_str = line.split()[2]
        model1_path = os.path.join('benchmark_models', f'{model1_str}.h5')
        model2_path = os.path.join('benchmark_models', f'{model2_str}.h5')
        model1 = Model(model1_path)
        model2 = Model(model2_path)
        comparison = ModelDiff(model1, model2)
        similarity = comparison.compare()
        eval_line = f'{model1_str} {model2_str} {similarity}'
        eval_lines.append(eval_line)
        print(eval_line)
    pathlib.Path('benchmark_models/model_pairs_eval.txt').write_text('\n'.join(eval_lines))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = parse_args()
    from benchmark import ImageBenchmark
    bench = ImageBenchmark(
        datasets_dir=os.path.join(args.benchmark_dir, 'data'),
        models_dir=os.path.join(args.benchmark_dir, 'models')
    )
    model1 = None
    model2 = None
    model_strs = []
    for model_wrapper in bench.list_models():
        if not model_wrapper.torch_model_exists():
            continue
        if model_wrapper.__str__() == args.model1:
            model1 = model_wrapper
        if model_wrapper.__str__() == args.model2:
            model2 = model_wrapper
        model_strs.append(model_wrapper.__str__())
    if model1 is None or model2 is None:
        print(f'model not found: {args.model1} {args.model2}')
        print(f'find models in the list:')
        print('\n'.join(model_strs))
        return
    comparison = ModelDiff(model1, model2)
    similarity = comparison.compare()
    print(f'the similarity is {similarity}')
    # evaluate_micro_benchmark()


if __name__ == '__main__':
    main()

