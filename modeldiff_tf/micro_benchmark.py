#!/usr/bin/python
# -*- coding: utf-8 -*-

# # Micro benchmark models trained on MNIST-like datasets
#
# ## image datasets
#
# - D_1 -- mnist
# - D_2 -- emnist
# - D_3 -- fashion_mnist
#
# ## base architectures
#
# - A_1 -- mlp
# - A_2 -- lenet
# - A_3 -- conv
#
# ## benchmark models
#
# ### Single transformations:
# - M_{i,x} -- A_i trained on D_x
# - M_{i,x}/{quant-dyn/int/flo} -- Compress M_{i,x} with dynamic range / integer / float16 quantization
# - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
# - M_{i,x}/{tune-q} -- Fine-tune M_{i,x} with q (ratio) new data
# - M_{i,x}/{trans-y,l} -- Transfer M_{i,x} to D_y by fine-tuning from l-st layer
# - M_{i,x}/{distil-j} -- Distill M_{i,x} to A_j
# - M_{i,x}/{steal-j,t} -- Steal M_{i,x} to A_j with dataset D_y
#
# ### Combinations:
# - M_{i,x}/{prune-p}/{tune-q} -- Prune and retrain
# - M_{i,x}/{trans-y,l}/{quant-dyn/int/flo} -- Transfer and quantize
# - M_{i,x}/{distil-j}/{trans-z,l} -- Distill and transfer


import os
import argparse
import logging
import pathlib
import random

from dataset import MyDataset
from model import MyModel, ModelGroup
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D, GlobalAveragePooling2D, Dropout, Softmax
from tensorflow.keras.models import Sequential


class MicroBenchmark:
    def __init__(self, models_dir):
        self.logger = logging.getLogger('MicroBench')
        self.datasets = {
            'MNIST': MyDataset('mnist'),
            'EMNIST': MyDataset('emnist/letters'),
            'FashionMNIST': MyDataset('fashion_mnist')
        }
        self.architectures = {
            'LeNet5': MicroBenchmark.LeNet5,
            'MLP': MicroBenchmark.MLP,
            'DeepCNN': MicroBenchmark.DeepCNN
        }
        self.model_group = ModelGroup(self.architectures, self.datasets, models_dir)
        self.models_dir = models_dir

    @staticmethod
    def LeNet5(input_shape, n_classes):
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(120, activation='tanh'))
        model.add(Dense(84, activation='tanh'))
        model.add(Dense(n_classes))
        model.add(Softmax())
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def MLP(input_shape, n_classes):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='sigmoid'))         # dense layer 1
        model.add(Dense(128, activation='sigmoid'))         # dense layer 2
        model.add(Dense(64, activation='sigmoid'))          # dense layer 3
        model.add(Dense(n_classes))
        model.add(Softmax())
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def DeepCNN(input_shape, n_classes):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(n_classes))
        model.add(Softmax())
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_model_variations(self, base_model):
        assert isinstance(base_model, MyModel)
        quantization_methods = ['integer', 'float16', 'dynamic']
        prune_ratios = [0.3, 0.6, 0.9]
        tune_ratios = [0.1, 0.5, 1]
        n_tune_layers_choices = [1, 2, 3]

        # - M_{i,x}/{quant-dyn/int/flo} -- Compress M_{i,x} with dynamic_range / integer / float16 quantization
        for quantization_method in quantization_methods:
            yield base_model.quantize(method=quantization_method)

        # - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
        for pr in prune_ratios:
            yield base_model.prune(prune_ratio=pr)

        # # - M_{i,x}/{tune-q} -- Fine-tune M_{i,x} with q (ratio) new data
        # for tune_ratio in tune_ratios:
        #     yield base_model.tune(tune_ratio=tune_ratio)

        # - M_{i,x}/{trans-y,l} -- Transfer M_{i,x} to D_y by fine-tuning from l-st layer
        for dataset_id in self.datasets:
            if dataset_id == base_model.dataset_id:
                continue
            for n_tune_layers in n_tune_layers_choices:
                yield base_model.transfer(dataset_id=dataset_id, n_tune_layers=n_tune_layers)

        # - M_{i,x}/{distill-j} -- Distill M_{i,x} to A_j
        for arch_id in self.architectures:
            yield base_model.distill(arch_id=arch_id)

        # - M_{i,x}/{steal-j,y} -- Steal M_{i,x} to A_j with dataset D_y
        for arch_id in self.architectures:
            for dataset_id in self.datasets:
                yield base_model.steal(arch_id=arch_id, dataset_id=dataset_id)

        # # Combinations:
        # # - M_{i,x}/{prune-p}/{tune-q} -- Prune and fine-tune
        # for prune_ratio in prune_ratios:
        #     for tune_ratio in tune_ratios:
        #         yield base_model.prune(prune_ratio=prune_ratio).tune(tune_ratio=tune_ratio)
        #
        # # - M_{i,x}/{trans-y,l}/{quant-dyn/int/flo} -- Transfer and quantize
        # for dataset_id in self.datasets:
        #     if dataset_id == base_model.dataset_id:
        #         continue
        #     for n_tune_layers in n_tune_layers_choices:
        #         for quantization_method in quantization_methods:
        #             yield base_model\
        #                 .transfer(dataset_id=dataset_id, n_tune_layers=n_tune_layers)\
        #                 .quantize(method=quantization_method)
        #
        # # - M_{i,x}/{distil-j}/{trans-z,l} -- Distill and transfer
        # for arch_id in self.architectures:
        #     for dataset_id in self.datasets:
        #         if dataset_id == base_model.dataset_id:
        #             continue
        #         for n_tune_layers in n_tune_layers_choices:
        #             yield base_model\
        #                 .distill(arch_id=arch_id)\
        #                 .transfer(dataset_id=dataset_id, n_tune_layers=n_tune_layers)

    def gen_models(self):
        model2variations = {}
        for arch_id in self.architectures:
            for dataset_id in self.datasets:
                base_model = self.model_group.train(arch_id, dataset_id, tag='v1', gen_if_not_exist=True)
                base_model_reinit = self.model_group.train(arch_id, dataset_id, tag='v2', gen_if_not_exist=True)
                variations = [base_model_reinit]
                for variation in self.get_model_variations(base_model=base_model):
                    self.logger.info(f'created model {variation.__str__()}')
                    variations.append(variation)
                model2variations[base_model] = variations
        all_models = []
        for model in model2variations:
            all_models.append(model)
            for variation in model2variations[model]:
                all_models.append(variation)
        self.model2variations = model2variations
        self.all_models = all_models
        return all_models

    def gen_model_pairs(self):
        """
        generate model pairs for comparison
        gen_models must be called before calling this method
        :return:
        """
        model_pairs = []
        model2variations = self.model2variations
        for base_model in model2variations:
            variations = model2variations[base_model]
            for variation in variations:
                model_pairs.append((base_model, variation))

            unrelated_variations = []
            for another_base_model in model2variations:
                if another_base_model == base_model:
                    continue
                for variation in model2variations[another_base_model]:
                    unrelated_variations.append(variation)
            unrelated_variations = random.sample(unrelated_variations, len(variations))
            for variation in unrelated_variations:
                model_pairs.append((base_model, variation))

        self.model_pairs = model_pairs
        return model_pairs


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Build micro benchmark.")

    parser.add_argument("-models_dir", action="store", dest="models_dir", default='benchmark_models',
                        help="Path to the benchmark model dir.")
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = parse_args()
    benchmark = MicroBenchmark(models_dir=args.models_dir)

    benchmark.gen_models()
    # print(benchmark.model2variations)

    model_pairs = benchmark.gen_model_pairs()
    # print(model_pairs)

    pair_lines = []
    for model1, model2 in model_pairs:
        print(f'model pair: {model1} {model2}')
        pair_line = f'{model1} \t {model1.accuracy} \t {model2} \t {model2.accuracy}'
        print(pair_line)
        pair_lines.append(pair_line)
    model_pairs_path = os.path.join(args.models_dir, 'model_pairs.txt')
    pathlib.Path(model_pairs_path).write_text('\n'.join(pair_lines))

