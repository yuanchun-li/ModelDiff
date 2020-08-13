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
import numpy as np
import tensorflow as tf
from scipy import spatial
from abc import ABC, abstractmethod

from utils import lazy_property, Utils


class Model:
    def __init__(self, model_path):
        assert model_path.endswith('.h5') or model_path.endswith('.tflite')
        self.logger = logging.getLogger('Model')
        self.model_path = model_path
        if self.model_path.endswith('h5'):
            self.tflite_path = self.model_path.replace('.h5', '.tflite')
            if not os.path.exists(self.tflite_path):
                keras_model = tf.keras.models.load_model(self.model_path)
                converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                tflite_content = converter.convert()
                pathlib.Path(self.tflite_path).write_bytes(tflite_content)
        else:
            self.tflite_path = self.model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        self.interpreter.allocate_tensors()

    @lazy_property
    def _input_details(self):
        return self.interpreter.get_input_details()

    @lazy_property
    def _output_details(self):
        return self.interpreter.get_output_details()

    @lazy_property
    def input_shape(self):
        return self._input_details[0]['shape']

    @lazy_property
    def output_shape(self):
        return self._output_details[0]['shape']

    @lazy_property
    def input_index(self):
        return self._input_details[0]['index']

    @lazy_property
    def output_index(self):
        return self._output_details[0]['index']

    @lazy_property
    def list_tensors(self):
        """
        return a list of tensors in the model
        """
        return self.interpreter.get_tensor_details()

    def get_seed_inputs(self, n):
        inputs = []
        from dataset import MyDataset
        ds = MyDataset('mnist')
        for images, labels in ds.test_ds.shuffle(n*10).batch(1).take(n):
            # print(images.shape)
            inputs.append(images)
        # for i in range(n):
        #     # inputs.append(np.random.normal(loc=0.5, size=self.input_shape).astype(np.float32))
        #     inputs.append(np.random.random_sample(self.input_shape).astype(np.float32))
        return inputs

    def forward(self, x):
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        y = self.interpreter.get_tensor(self.output_index)
        return y

    def batch_forward(self, batch_inputs):
        outputs = []
        for i in list(batch_inputs):
            output = self.forward(i)
            outputs.append(output)
        batch_outputs = np.concatenate(outputs)
        return batch_outputs

    def get_tensor_value(self, tensor):
        """
        get the value of a tensor
        """
        return self.interpreter.get_tensor(tensor['index'])

    def get_tensor_values(self):
        """
        get a list of tensors together with their values
        """
        tensor_values = []
        for tensor in self.list_tensors:
            tensor_value = self.get_tensor_value(tensor)
            tensor_values.append((tensor, tensor_value))
        return tensor_values


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
        self.logger.debug(f'initialize comparison: {self.model1} {self.model2}')
        self.logger.debug(f'input shapes: {self.model1.input_shape} {self.model2.input_shape}')
        self.input_shape = model1.input_shape
        if list(model1.input_shape) != list(model2.input_shape):
            self.logger.warning('input shapes do not match')
        self.gen_inputs = gen_inputs if gen_inputs else ModelDiff._gen_profiling_inputs_search
        self.input_metrics = input_metrics if input_metrics else ModelDiff.metrics_output_range
        self.compute_decision_dist = compute_decision_dist if compute_decision_dist else ModelDiff._compute_decision_dist_output_cos
        self.compare_ddv = compare_ddv if compare_ddv else ModelDiff._compare_ddv_cos

    def compare(self):
        self.logger.info(f'generating profiling inputs')
        seed_inputs = self.model1.get_seed_inputs(self.N_INPUT_PAIRS) + self.model2.get_seed_inputs(self.N_INPUT_PAIRS)
        random.shuffle(seed_inputs)
        inputs = list(self.gen_inputs(self, seed_inputs))
        batch_inputs = np.array(inputs)
        batch_outputs_1 = self.model1.batch_forward(batch_inputs)
        batch_outputs_2 = self.model2.batch_forward(batch_inputs)
        self.logger.info(f'batch_outputs_1={batch_outputs_1}\nbatch_outputs_2={batch_outputs_2}')

        input_pairs = []
        for i in range(int(len(inputs) / 2)):
            xa = inputs[2 * i]
            xb = inputs[2 * i + 1]
            input_pairs.append((xa, xb))

        input_metrics_1 = self.input_metrics(self.model1, inputs)
        input_metrics_2 = self.input_metrics(self.model2, inputs)
        self.logger.info(f'input metrics: model1={input_metrics_1} model2={input_metrics_2}')

        self.logger.info(f'computing DDVs')
        ddv1 = []  # DDV is short for decision distance vector
        ddv2 = []
        for i, (xa, xb) in enumerate(input_pairs):
            # self.logger.info(f'generated input pair:\n{xa}\n{xb}')
            dist1 = self.compute_decision_dist(self.model1, xa, xb)
            dist2 = self.compute_decision_dist(self.model2, xa, xb)
            # self.logger.debug(f'computed distances: {dist1} {dist2}')
            ddv1.append(dist1)
            ddv2.append(dist2)

        self.logger.info(f'comparing DDVs')
        ddv1 = Utils.normalize(np.array(ddv1))
        ddv2 = Utils.normalize(np.array(ddv2))
        self.logger.info(f'ddv1={ddv1}\nddv2={ddv2}')
        # self.logger.debug(f'model1 ddv:\n{ddv1}')
        # self.logger.debug(f'model2 ddv:\n{ddv2}')
        ddv_distance = self.compare_ddv(ddv1, ddv2)
        model_similarity = 1 - ddv_distance
        self.logger.info(f'model similarity is {model_similarity}')
        return model_similarity

    def test_input_generators(self):
        seed_inputs = self.model1.get_seed_inputs(self.N_INPUT_PAIRS) + self.model2.get_seed_inputs(self.N_INPUT_PAIRS)
        inputs = ModelDiff._gen_profiling_inputs_none(self, seed_inputs)

    @staticmethod
    def metrics_output_diversity(model, inputs):
        outputs = []
        for i in inputs:
            output = model.forward(i)
            outputs.append(output)
        output_dists = []
        for i in range(0, len(outputs) - 1):
            for j in range(i + 1, len(outputs)):
                output_dist = spatial.distance.euclidean(outputs[i], outputs[j])
                output_dists.append(output_dist)
        diversity = sum(output_dists) / len(output_dists)
        return diversity

    @staticmethod
    def metrics_output_variance(model, inputs):
        outputs = []
        for i in inputs:
            output = model.forward(i)
            outputs.append(output)
        batch_output = np.array(outputs)

        # print(batch_output.shape)
        mean_axis = tuple(list(range(len(batch_output.shape)))[1:-1])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        # print(batch_output_mean.shape)
        output_variances = np.var(batch_output_mean, axis=0)
        # print(output_variances)
        return np.prod(output_variances)

    @staticmethod
    def metrics_output_range(model, inputs):
        outputs = []
        for i in inputs:
            output = model.forward(i)
            outputs.append(output)
        batch_output = np.array(outputs)

        # print(batch_output.shape)
        mean_axis = tuple(list(range(len(batch_output.shape)))[1:-1])
        batch_output_mean = np.mean(batch_output, axis=mean_axis)
        # print(batch_output_mean.shape)
        output_ranges = np.max(batch_output_mean, axis=0) - np.min(batch_output_mean, axis=0)
        # print(output_variances)
        return np.prod(output_ranges)

    @staticmethod
    def metrics_neuron_coverage(model, inputs):
        outputs = []
        tensor_values_list = []
        # print(model.list_tensors)
        for i in inputs:
            output = model.forward(i)
            outputs.append(output)
            tensor_values = model.get_tensor_values()
            tensor_values_list.append(tensor_values)
        batch_tensor_values = []
        for i, (tensor, _) in enumerate(tensor_values_list[0]):
            batch_tensor_value = []
            for tensor_values in tensor_values_list:
                # print(f'{tensor["name"]} - {tensor_values[i][0]["name"]}')
                assert tensor["index"] == tensor_values[i][0]["index"]
                batch_tensor_value.append(tensor_values[i][1])
            batch_tensor_value = np.array(batch_tensor_value)
            batch_tensor_values.append((tensor, batch_tensor_value))

        neurons = []
        neurons_covered = []
        for tensor, batch_tensor_value in batch_tensor_values:
            # print(f'{tensor["name"]} {batch_tensor_value.shape}')
            # if 'relu' not in tensor["name"].lower():
            #     continue
            squeeze_axis = tuple(list(range(len(batch_tensor_value.shape)))[:-1])
            squeezed_tensor_value = np.max(batch_tensor_value, axis=squeeze_axis)
            for i in range(batch_tensor_value.shape[-1]):
                neuron_name = f'{tensor["name"]}-{i}'
                neurons.append(neuron_name)
                neuron_value = squeezed_tensor_value[i]
                covered = neuron_value > 0.1
                if covered:
                    neurons_covered.append(neuron_name)
        neurons_not_covered = [neuron for neuron in neurons if neuron not in neurons_covered]
        print(f'{len(neurons_not_covered)} neurons_not_covered: {neurons_not_covered}')
        return float(len(neurons_covered)) / len(neurons)

    @staticmethod
    def _compute_decision_dist_output_cos(model, xa, xb):
        ya = model.forward(xa)
        yb = model.forward(xb)
        return spatial.distance.cosine(ya, yb)

    # def gen_profiling_inputs(self, n):
    #     inputs = list(self._gen_profiling_inputs_random(n))
    #     diversity1 = self.model1.get_diversity(inputs)
    #     diversity2 = self.model2.get_diversity(inputs)
    #     print(f'_gen_profiling_inputs_random inputs diversity: {diversity1} {diversity2}')
    #     inputs = list(self._gen_profiling_inputs_same(n))
    #     diversity1 = self.model1.get_diversity(inputs)
    #     diversity2 = self.model2.get_diversity(inputs)
    #     print(f'_gen_profiling_inputs_same inputs diversity: {diversity1} {diversity2}')
    #     inputs = list(self._gen_profiling_inputs_1pixel(n))
    #     diversity1 = self.model1.get_diversity(inputs)
    #     diversity2 = self.model2.get_diversity(inputs)
    #     print(f'_gen_profiling_inputs_1pixel inputs diversity: {diversity1} {diversity2}')
    #     return inputs

    @staticmethod
    def _gen_profiling_inputs_none(comparator, seed_inputs):
        return seed_inputs

    @staticmethod
    def _gen_profiling_inputs_random(comparator, seed_inputs):
        input_shape = seed_inputs[0].shape
        inputs = []
        for i in range(len(seed_inputs)):
            inputs.append(np.random.normal(size=input_shape).astype(np.float32))
        return inputs

    # @staticmethod
    #     # def _gen_profiling_inputs_1pixel(comparator, seed_inputs):
    #     #     input_shape = seed_inputs[0].shape
    #     #     for i in range(len(seed_inputs)):
    #     #         x = np.zeros(input_shape, dtype=np.float32)
    #     #         random_index = np.unravel_index(np.argmax(np.random.normal(size=input_shape)), input_shape)
    #     #         x[random_index] = 1
    #     #         yield x

    @staticmethod
    def _gen_profiling_inputs_search(comparator, seed_inputs):
        batch_seed_inputs = np.array(seed_inputs)
        inputs = batch_seed_inputs
        input_shape = seed_inputs[0].shape
        max_iterations = 10
        max_steps = 10

        initial_outputs_1 = comparator.model1.batch_forward(batch_seed_inputs)
        initial_outputs_2 = comparator.model2.batch_forward(batch_seed_inputs)

        for i in range(max_iterations):
            metrics = comparator.input_metrics(comparator.model1, list(inputs))
            # mutation_idx = random.randint(0, len(inputs))
            mutation_perturbation = np.random.random_sample(size=batch_seed_inputs.shape).astype(np.float32)
            # print(f'{inputs.shape} {mutation_perturbation.shape}')
            print(f'mutation {i}-th iteration')
            for j in range(max_steps):
                mutated_inputs = inputs + mutation_perturbation
                # print(f'{list(inputs)[0].shape}')
                mutated_metrics = comparator.input_metrics(comparator.model1, list(inputs))
                print(f'  metrics: {metrics} -> {mutated_metrics}')
                # if mutated_metrics <= metrics:
                #     break
                metrics = mutated_metrics
                inputs = mutated_inputs
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
        if model_wrapper.__str__() in args.model1:
            model1 = model_wrapper
        if model_wrapper.__str__() in args.model2:
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
