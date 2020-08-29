#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import lazy_property
from pdb import set_trace as st

BATCH_SIZE = 128


class MyDataset1:
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

from tensorflow.keras.preprocessing import image

import numpy as np
import random
np.random.seed(3)
random.seed(3)

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
        self.curr_idx = 0

    def _load_dataset(self):
        file_names = os.listdir(path='/home/yuancli/data/imagenet_samples')
        self.file_path = [os.path.join('/home/yuancli/data/imagenet_samples', name) for name in file_names]
        
    def sample(self, size=50):
        # self.sample_path = random.sample(self.file_path, size)
        self.sample_path = self.file_path[self.curr_idx:self.curr_idx+size]
        self.curr_idx += size
        if self.curr_idx >= len(self.file_path):
            self.curr_idx = 0

        image_list = []
        for i in range(size):
            img = image.load_img(self.sample_path[i], target_size=(224, 224))
            x = image.img_to_array(img)
            # x = x / 255 * 2 - 1
            x = np.expand_dims(x, axis=0)
            image_list.append(x)
            print(self.sample_path[i])
        # sample = np.concatenate(image_list)
        return image_list

    def get_sample_path(self):
        return self.sample_path

if __name__=="__main__":
        
    d = MyDataset("Imagenet")
    d.sample()
