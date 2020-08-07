#!/usr/bin/python
# -*- coding: utf-8 -*-

# # Benchmark models trained on image datasets
#
# ## image datasets
#
# Teacher dataset: ImageNet
# Teacher dataset subset: imagenette
#
# Student datasets:
# - D_1 -- oxford_flowers102  https://www.tensorflow.org/datasets/catalog/oxford_flowers102
# - D_2 -- standard_dogs  https://www.tensorflow.org/datasets/catalog/stanford_dogs
# - D_3 -- food101  https://www.tensorflow.org/datasets/catalog/food101
#
# ## base architectures
#
# - A_1 -- ResNet50        98 MB, 25,636,712 params, 0.749
# - A_2 -- DenseNet169     57 MB, 14,307,880 params, 0.762
# - A_3 -- MobileNetV2     14 MB, 3,538,984 params, 0.713
#
# ## benchmark models
#
# ### Single transformations:
# - M_{i,x} -- A_i trained on D_x, load pretrained models from public APIs
# - M_{i,x} -- another A_i trained on D_x, for reference
# - M_{i,x}/{quant-dyn/int/flo} -- Compress M_{i,x} with dynamic range / integer / float16 quantization
# - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
# - M_{i,x}/{trans-y,l} -- Transfer M_{i,x} to D_y by fine-tuning last l& layers
# - M_{i,x}/{distil-j} -- Distill M_{i,x} to A_j
# - M_{i,x}/{steal-j,t} -- Steal M_{i,x} to A_j with dataset D_y


import os
import argparse
import logging
import pathlib
import random
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D, GlobalAveragePooling2D, Dropout, Softmax, Concatenate
from tensorflow.keras.models import Sequential, Model
from utils import lazy_property, Utils


INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 128
TRAIN_EPOCHS = 5
TUNE_EPOCHS = 30
PRUNE_EPOCHS = 3
DISTILL_EPOCHS = 5


class ImagenetLikeDataset:
    def __init__(self, name):
        self.name = name
        self.input_shape = INPUT_SHAPE
        self.output_shape = None
        self.n_classes = None
        self.dataset = None
        self.info = None
        self._load_dataset()

    def _load_dataset(self):
        self.dataset, self.info = tfds.load(self.name, shuffle_files=True, with_info=True, as_supervised=True)
        self.input_shape = self.info.features['image'].shape
        self.n_classes = self.info.features['label'].num_classes
        self.output_shape = self.n_classes

    def _normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=(224,224))
        return image, label

    @lazy_property
    def train_ds(self):
        return self.dataset['train'] \
            .map(self._normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .cache()

    @lazy_property
    def test_ds(self):
        return self.dataset['test'] \
            .map(self._normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .cache()

    @lazy_property
    def train_ds_count(self):
        return len(list(self.train_ds))

    @lazy_property
    def test_ds_count(self):
        return len(list(self.test_ds))


class ModelGroup:
    def __init__(self, architectures, datasets, models_dir):
        self.architectures = architectures
        self.datasets = datasets
        self.models_dir = models_dir

    def train(self, arch_id, tag='pretrain', epochs=TRAIN_EPOCHS):
        dataset_id = 'imagenet'
        model = MyModel(
            model_group=self,
            base_model=None,
            transformation_str=f'model({arch_id},{dataset_id})',
            arch_id=arch_id,
            dataset_id=dataset_id,
            tag=tag
        )
        if not model.keras_exists():
            arch = self.architectures[arch_id]
            if tag == 'pretrain':
                keras_model = arch(input_shape=INPUT_SHAPE, weights='imagenet')
                eras_model = Utils.get_compiled_model(keras_model)
            else:
                model.logger.info(f'training: {model.__str__()}')
                dataset = ImagenetLikeDataset('imagenet_v2')
                train_ds = dataset.test_ds.batch(BATCH_SIZE)
                # for images, labels in train_ds:
                #     Utils.show_images(images, labels)
                #     continue
                keras_model = arch(input_shape=INPUT_SHAPE, weights=None)
                print(keras_model.summary())
                keras_model = Utils.get_compiled_model(keras_model)
                keras_model.fit(train_ds, epochs=epochs)
            keras_model.save(model.keras_model_path)
        return model


class MyModel:
    def __init__(self, model_group, base_model, transformation_str,
                 arch_id=None, dataset_id=None, eval_dataset_id=None, gen_if_not_exist=True, tag=None):
        self.logger = logging.getLogger('MyModel')
        self.model_group = model_group
        self.base_model = base_model
        self.transformation_str = transformation_str
        self.arch_id = arch_id if arch_id else base_model.arch_id
        self.dataset_id = dataset_id if dataset_id else base_model.dataset_id
        self.eval_dataset_id = eval_dataset_id if eval_dataset_id else self.dataset_id
        self.gen_if_not_exist = base_model.gen_if_not_exist if base_model else gen_if_not_exist
        self.tag = tag if tag else base_model.tag

        assert self.tag is not None
        assert self.arch_id is not None
        assert self.dataset_id is not None
        assert self.eval_dataset_id is not None

        self.keras_model_path = os.path.join(model_group.models_dir, f'{self.__str__()}.h5')
        self.tflite_model_path = os.path.join(model_group.models_dir, f'{self.__str__()}.tflite')

    def __str__(self):
        teacher_str = self.tag if self.base_model is None else self.base_model.__str__()
        return f'{teacher_str}-{self.transformation_str}'

    def keras_exists(self):
        return os.path.exists(self.keras_model_path)

    def tflite_exists(self):
        return os.path.exists(self.tflite_model_path)

    @lazy_property
    def keras_model(self):
        return tf.keras.models.load_model(self.keras_model_path)

    @lazy_property
    def tflite_interpreter(self):
        if not self.tflite_exists():
            self.convert_keras_to_tflite()
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        return interpreter

    def convert_keras_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model_file(self.keras_model_path)
        tflite_model = converter.convert()
        pathlib.Path(self.tflite_model_path).write_bytes(tflite_model)

    def quantize(self, method='integer'):
        """
        do post-training quantization on the model
        :param method: one of integer, float16, and dynamic
        :return:
        """
        trans_str = f'quantize({method})'
        model = MyModel(
            model_group=self.model_group,
            base_model=self,
            transformation_str=trans_str
        )
        if not model.tflite_exists() and self.gen_if_not_exist:
            model.logger.info(f'generating: {model.__str__()}')
            original_model = self.keras_model
            converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
            if method == 'integer':
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

                def representative_data_gen():
                    for images, labels in self.model_group.datasets[self.dataset_id].train_ds.batch(1).take(100):
                        yield [images]

                converter.representative_dataset = representative_data_gen
            elif method == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            pathlib.Path(model.tflite_model_path).write_bytes(tflite_model)
        return model

    def prune(self, prune_ratio=0.1, epochs=PRUNE_EPOCHS):
        trans_str = f'prune({prune_ratio})'
        model = MyModel(
            model_group=self.model_group,
            transformation_str=trans_str,
            base_model=self
        )
        if not model.keras_exists() and self.gen_if_not_exist:
            model.logger.info(f'generating: {model.__str__()}')
            original_model = self.keras_model
            # _, keras_file = tempfile.mkstemp('.h5')
            # tf.keras.models.save_model(original_model, keras_file, include_optimizer=False)
            # original_model = tf.keras.models.load_model(keras_file)
            # print('Saved baseline model to:', keras_file)

            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                original_model, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(prune_ratio, 0)
            )

            log_dir = tempfile.mkdtemp()
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                # Log sparsity and other metrics in Tensorboard.
                tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
            ]
            dataset = self.model_group.datasets[self.dataset_id]
            train_ds = dataset.train_ds.batch(BATCH_SIZE)
            keras_model = Utils.get_compiled_model(model_for_pruning)
            keras_model.fit(train_ds, callbacks=callbacks, epochs=epochs)

            model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
            # print("final model")
            # model_for_export.summary()
            model_for_export.save(model.keras_model_path)
        return model

    def transfer(self, dataset_id, tune_ratio=0.1, epochs=TUNE_EPOCHS):
        trans_str = f'transfer({dataset_id},{tune_ratio})'
        model = MyModel(
            model_group=self.model_group,
            transformation_str=trans_str,
            base_model=self,
            dataset_id=dataset_id
        )
        if not model.keras_exists() and self.gen_if_not_exist:
            model.logger.info(f'generating: {model.__str__()}')
            dataset = self.model_group.datasets[model.dataset_id]
            keras_model = self._do_transfer(self.keras_model, dataset, tune_ratio, epochs)
            keras_model.save(model.keras_model_path)
        return model

    def _do_transfer(self, original_model, dataset, tune_ratio=0.1, epochs=TUNE_EPOCHS):
        assert tune_ratio > 0
        # n_tune_layers = n_tune_layers - 1
        arch = self.model_group.architectures[self.arch_id]
        base_model = arch(input_shape=INPUT_SHAPE, include_top=False, weights=None)
        Utils.copy_weights(original_model, base_model)
        transfer_pool = GlobalAveragePooling2D(name='transfer_pool')
        transfer_dense = Dense(dataset.n_classes, name='transfer_dense')
        model = Sequential([
            base_model,
            transfer_pool,
            transfer_dense
        ])

        # base_model = tf.keras.models.clone_model(original_model)
        # conv_dense_layers = []
        # for layer in base_model.layers:
        #     if isinstance(layer, Dense) or isinstance(layer, Conv2D):
        #         conv_dense_layers.append(layer)
        # transfer_layer = conv_dense_layers[-2]
        # base_model = Model(inputs=base_model.input, outputs=transfer_layer.output)
        # base_model.set_weights(original_model.get_weights())
        # Utils.copy_weights(original_model, base_model)

        # if isinstance(transfer_layer, Dense):
        #    transfer_dense = Dense(dataset.n_classes, name='transfer_dense')
        #    model = Sequential([
        #         *base_model.layers,
        #         transfer_dense
        #     ])
        # else:
        #     transfer_pool = GlobalAveragePooling2D(name='transfer_pool')
        #     transfer_dense = Dense(dataset.n_classes, name='transfer_dense')
        #     model = Sequential([
        #         *base_model.layers,
        #         transfer_pool,
        #         transfer_dense
        #     ])

        weight_layers = []
        for layer in base_model.layers:
            if layer.get_weights():
                weight_layers.append(layer)
        n_tune_layers = max(int(len(weight_layers) * tune_ratio), 1)
        tune_layers = weight_layers[-n_tune_layers:]

        base_model.trainable = True
        for layer in base_model.layers:
            if layer in tune_layers:
                break
            layer.trainable = False

        model = Utils.get_compiled_model(model)
        print(model.summary())
        train_ds = dataset.train_ds.batch(BATCH_SIZE)
        model.fit(train_ds, epochs=epochs)
        return model

    def distill(self, arch_id, epochs=DISTILL_EPOCHS):
        trans_str = f'distill({arch_id})'
        model = MyModel(
            model_group=self.model_group,
            base_model=self,
            transformation_str=trans_str,
            arch_id=arch_id
        )
        if not model.keras_exists() and self.gen_if_not_exist:
            model.logger.info(f'generating: {model.__str__()}')
            keras_model = self._do_distill(
                teacher_model=self.keras_model,
                student_arch=self.model_group.architectures[arch_id],
                dataset=self.model_group.datasets[model.dataset_id],
                epochs=epochs
            )
            keras_model.save(model.keras_model_path)
        return model

    def _do_distill(self, teacher_model, student_arch, dataset, temperature=3.0, lambda_const=0.5,
                    epochs=DISTILL_EPOCHS):
        # References:
        # https://github.com/Ujjwal-9/Knowledge-Distillation
        # https://github.com/johnkorn/distillation/blob/master/train.py
        assert isinstance(teacher_model.layers[-1], Softmax)
        n_classes = dataset.n_classes
        original_teacher_model = teacher_model
        teacher_model = Model(inputs=teacher_model.input, outputs=teacher_model.layers[-2].output)
        # teacher_model.set_weights(original_teacher_model.get_weights())
        Utils.copy_weights(original_teacher_model, teacher_model)

        # print('teacher model:')
        # print(teacher_model.summary())

        x_train = []
        y_train = []
        for images, labels in dataset.train_ds.batch(BATCH_SIZE).cache():
            # y = tf.keras.utils.to_categorical(labels, num_classes=n_classes)
            y = tf.one_hot(labels, n_classes)
            teacher_logits = teacher_model.predict_on_batch(images)
            # teacher_predictions = tf.nn.softmax(teacher_logits)
            # print(y, teacher_logits, teacher_predictions)
            y = tf.concat([y, teacher_logits], axis=1)
            x_train.append(images)
            y_train.append(y)
        x_train = tf.concat(x_train, axis=0)
        y_train = tf.concat(y_train, axis=0)
        # print(x_train.shape, y_train.shape)

        # build student model
        model = student_arch(dataset.input_shape, dataset.output_shape)
        # model.layers.pop()
        # usual probabilities
        logits = model.layers[-2].output
        probabilities = Softmax()(logits)

        # # softed probabilities
        # logits_T = Lambda(lambda x: x / temperature)(logits)
        # probabilities_T = Softmax()(logits_T)

        output = Concatenate()([probabilities, logits])
        model = Model(model.input, output)

        # print('student model:')
        # print(model.summary())

        def knowledge_distillation_loss(y_true, y_pred):
            y_true, teacher_logits = y_true[:, :n_classes], y_true[:, n_classes:]
            y_pred, student_logits = y_pred[:, :n_classes], y_pred[:, n_classes:]

            y_soft = tf.keras.backend.softmax(teacher_logits / temperature)
            y_pred_soft = tf.keras.backend.softmax(student_logits / temperature)

            return lambda_const * tf.losses.categorical_crossentropy(y_true, y_pred) \
                   + tf.losses.categorical_crossentropy(y_soft, y_pred_soft)
            # return tf.losses.kld(y_soft, y_pred_soft) * temperature * temperature
            # return tf.losses.categorical_crossentropy(y_soft, y_pred_soft)

        def acc(y_true, y_pred):
            y_true = y_true[:, :n_classes]
            y_pred = y_pred[:, :n_classes]
            return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

        def soft_logloss(y_true, y_pred):
            y_true, teacher_logits = y_true[:, :n_classes], y_true[:, n_classes:]
            y_pred, student_logits = y_pred[:, :n_classes], y_pred[:, n_classes:]
            y_soft = tf.keras.backend.softmax(teacher_logits / temperature)
            y_pred_soft = tf.keras.backend.softmax(student_logits / temperature)
            return tf.losses.categorical_crossentropy(y_soft, y_pred_soft)

        model.compile(
            optimizer='adam',
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred),
            metrics=[acc, soft_logloss]
        )

        model.fit(x_train, y_train, epochs=epochs)

        distilled_model = Model(inputs=model.input, outputs=probabilities)
        Utils.copy_weights(model, distilled_model)
        distilled_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # distilled_model.evaluate(dataset.test_ds.batch(BATCH_SIZE))
        # print('distilled model:')
        # print(distilled_model.summary())
        return distilled_model

    def steal(self, arch_id, dataset_id=None, epochs=DISTILL_EPOCHS):
        if dataset_id is None:
            dataset_id = self.dataset_id
        trans_str = f'steal({arch_id},{dataset_id})'
        model = MyModel(
            model_group=self.model_group,
            transformation_str=trans_str,
            base_model=self,
            arch_id=arch_id,
            dataset_id=dataset_id,
            eval_dataset_id=self.dataset_id
        )
        if not model.keras_exists() and self.gen_if_not_exist:
            model.logger.info(f'generating: {model.__str__()}')
            keras_model = self._do_steal(
                teacher_model=self.keras_model,
                teacher_dataset=self.model_group.datasets[self.dataset_id],
                student_arch=self.model_group.architectures[arch_id],
                student_dataset=self.model_group.datasets[dataset_id],
                epochs=epochs
            )
            keras_model.save(model.keras_model_path)
        return model

    def _do_steal(self, teacher_model, teacher_dataset, student_arch, student_dataset, epochs=DISTILL_EPOCHS):
        assert isinstance(teacher_model.layers[-1], Softmax)
        n_classes = teacher_dataset.n_classes
        x_train = []
        y_train = []
        for images, labels in student_dataset.train_ds.batch(BATCH_SIZE).cache():
            one_hot_labels = tf.one_hot(labels, n_classes)
            teacher_pred = teacher_model.predict_on_batch(images)
            y = tf.concat([one_hot_labels, teacher_pred], axis=1)
            x_train.append(images)
            y_train.append(y)
        x_train = tf.concat(x_train, axis=0)
        y_train = tf.concat(y_train, axis=0)
        # print(x_train.shape, y_train.shape)

        # build student model
        model = student_arch(teacher_dataset.input_shape, teacher_dataset.output_shape)

        def steal_loss(y_true, y_pred):
            one_hot_labels, teacher_pred = y_true[:, :n_classes], y_true[:, n_classes:]
            return tf.keras.losses.categorical_crossentropy(teacher_pred, y_pred)

        def acc(y_true, y_pred):
            one_hot_labels, teacher_pred = y_true[:, :n_classes], y_true[:, n_classes:]
            return tf.keras.metrics.categorical_accuracy(one_hot_labels, y_pred)

        model.compile(
            optimizer='adam',
            loss=steal_loss,
            metrics=[acc]
        )

        model.fit(x_train, y_train, epochs=epochs)
        stolen_model = Model(inputs=model.input, outputs=model.output)
        Utils.copy_weights(model, stolen_model)
        stolen_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        stolen_model.evaluate(teacher_dataset.test_ds.batch(BATCH_SIZE))
        return stolen_model

    @lazy_property
    def accuracy(self):
        test_ds = self.model_group.datasets[self.eval_dataset_id].test_ds.batch(64)
        if self.keras_exists():
            try:
                results = self.keras_model.evaluate(test_ds)
                return results[1]
            except:
                self.logger.warning('cannot measure keras model accuracy')
        try:
            return self._evaluate_tflite(self.tflite_interpreter, test_ds)
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Interrupted by user')
        except:
            self.logger.error('cannot measure model accuracy')
            return 0

    def _evaluate_tflite(self, interpreter, test_ds):
        accurate_count = 0

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Run predictions on every image in the "test" dataset.
        predictions = []
        for (val_images, val_labels) in test_ds:
            for val_image, val_label in zip(val_images, val_labels):
                val_image = tf.expand_dims(val_image, 0)
                interpreter.set_tensor(input_index, val_image)

                # Run inference.
                interpreter.invoke()

                # Post-processing: remove batch dimension and find the digit with highest probability.
                probability = interpreter.get_tensor(output_index)
                predicted_label = np.argmax(probability[0])
                predictions.append(predicted_label)

                # Compare prediction results with ground truth labels to calculate accuracy.
                if predicted_label == val_label:
                    accurate_count += 1

        accuracy = accurate_count * 1.0 / len(predictions)
        return accuracy


class ImageBenchmark:
    def __init__(self, models_dir):
        self.logger = logging.getLogger('ImageBench')
        # self.imagenet_v2 = ImagenetLikeDataset('imagenet_v2')
        self.datasets = {
            'oxford_flowers102': ImagenetLikeDataset('oxford_flowers102'),
            'stanford_dogs': ImagenetLikeDataset('stanford_dogs'),
            'food101': ImagenetLikeDataset('food101')
        }
        self.architectures = {
            'MobileNetV2': tf.keras.applications.MobileNetV2,
            'DenseNet169': tf.keras.applications.DenseNet169,
            'ResNet50': tf.keras.applications.ResNet50
        }
        self.model_group = ModelGroup(self.architectures, self.datasets, models_dir)
        self.models_dir = models_dir

    def get_model_variations(self, base_model):
        assert isinstance(base_model, MyModel)
        quantization_methods = ['integer', 'float16', 'dynamic']
        prune_ratios = [0.2, 0.5, 0.8]
        transfer_tune_ratios = [0.1, 0.5, 1]

        transfer_models = []
        # - M_{i,x}/{trans-y,l} -- Transfer M_{i,x} to D_y by fine-tuning from l-st layer
        for dataset_id in self.datasets:
            if dataset_id == base_model.dataset_id:
                continue
            for tune_ratio in transfer_tune_ratios:
                transfer_model = base_model.transfer(dataset_id=dataset_id, tune_ratio=tune_ratio)
                transfer_models.append(transfer_model)
                yield transfer_model

        # - M_{i,x}/{quant-dyn/int/flo} -- Compress M_{i,x} with dynamic_range / integer / float16 quantization
        for transfer_model in transfer_models:
            for quantization_method in quantization_methods:
                yield transfer_model.quantize(method=quantization_method)

        # - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
        for transfer_model in transfer_models:
            for pr in prune_ratios:
                yield transfer_model.prune(prune_ratio=pr)

        # # - M_{i,x}/{distill-j} -- Distill M_{i,x} to A_j
        # for arch_id in self.architectures:
        #     yield base_model.distill(arch_id=arch_id)

        # - M_{i,x}/{steal-j} -- Steal M_{i,x} to A_j
        for transfer_model in transfer_models:
            for arch_id in self.architectures:
                yield transfer_model.steal(arch_id=arch_id)

    def gen_models(self):
        model2variations = {}
        for arch_id in self.architectures:
            base_model = self.model_group.train(arch_id, tag='pretrain')
            variations = []
            for variation in self.get_model_variations(base_model=base_model):
                self.logger.info(f'created model {variation.__str__()}')
                variations.append(variation)
            base_model_reinit = self.model_group.train(arch_id, tag='retrain')
            # variations = [base_model_reinit]
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
    benchmark = ImageBenchmark(models_dir=args.models_dir)

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

