#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import lazy_property

BATCH_SIZE = 128


class MyDataset:
    def __init__(self, name):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.n_classes = None
        self.n_examples = 0
        self.dataset = None
        self.info = None
        self._load_dataset()

    def _load_dataset(self):
        if 'mnist' in self.name:
            self.dataset, self.info = tfds.load(self.name, shuffle_files=True, with_info=True, as_supervised=True)
            self.input_shape = self.info.features['image'].shape
            self.n_classes = self.info.features['label'].num_classes
            self.n_examples = self.info
            self.output_shape = self.n_classes
        # self.input_shape = dataset.info.features

    def _normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    def transform(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, 56, 56)
        image = tf.image.resize(image, size=[28, 28])
        image = tf.image.adjust_brightness(image, 0.5)
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
    def train_ds_transformed(self):
        return self.train_ds.map(self.transform)

    @lazy_property
    def test_ds_transformed(self):
        return self.test_ds.map(self.transform)

    @lazy_property
    def train_ds_count(self):
        return len(list(self.train_ds))

    @lazy_property
    def test_ds_count(self):
        return len(list(self.test_ds))

    def get_compiled_model(self, keras_model):
        keras_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return keras_model

