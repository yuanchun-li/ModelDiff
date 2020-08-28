#!/usr/bin/env python3
import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM
from foolbox import Misclassification

from pretrain_models import pretrain_models
from dataset import MyDataset

import numpy as np
import random

import shutil
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

if __name__ == "__main__":
    # instantiate a model
    for model_name in pretrain_models:
        print(model_name)
        model = pretrain_models[model_name]
        # model = tf.keras.applications.ResNet50(weights="imagenet")
        # pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
        # bounds = (0, 255)
        pre = dict()
        bounds = (-1, 1)
        fmodel = TensorFlowModel(model, bounds, preprocessing=pre)

        # get data and test the model
        # wrapping the tensors with ep.astensors is optional, but it allows
        # us to work with EagerPy tensors in the following
        dataset = MyDataset("Imagenet")
        sampled_data = dataset.sample(100)
        # images = ep.astensor(tf.convert_to_tensor(np.vstack(sampled_data)))
        images = ep.astensor(tf.convert_to_tensor(preprocess_input(np.concatenate(sampled_data))))
        labels = ep.astensor(tf.convert_to_tensor(np.stack(random.sample(range(0, 1000), 100))))
        print(type(images))
        print(images.dtype)
        print(images.dtype)
        print(type(labels))
        print(images.dtype)
        print("min", np.min(images.numpy()), "max", np.max(images.numpy()))
        """
        images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=100))
        print("min", np.min(images.numpy()), "max", np.max(images.numpy()))
        print(type(images))
        print(images.dtype)
        print(type(labels))
        print(images.dtype)
        """
        # print(accuracy(fmodel, images, labels))
        predictions = fmodel(images).argmax(axis=-1)
        print(predictions.numpy())
        test_predictions = model.predict(preprocess_input(np.concatenate(sampled_data)))
        print([x.argmax(axis=-1) for x in test_predictions])
        accuracy = (predictions == labels).float32().mean()
        print("random accuracy", accuracy.item())
        accuracy = (predictions == predictions).float32().mean()
        print("perfect accuracy", accuracy.item())

        # apply the attack
        attack = LinfPGD(abs_stepsize=1/255*2, steps=10)
        epsilons = [8/255*2]
        advs, _, success = attack(fmodel, images, Misclassification(labels), epsilons=epsilons)

        print(type(advs[0][0].numpy()))
        print(advs[0][0].numpy().shape)

        advs_images = ep.stack(advs[0])
        print(advs_images.shape)
        advs_predictions = fmodel(advs_images).argmax(axis=-1)
        advs_accuracy = (predictions == advs_predictions).float32().mean()
        print("two predictions overlap", advs_accuracy.item())

        output_dir = "advs_images_tf_%s" % model_name
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        sample_path = dataset.get_sample_path()
        for p, img in zip(sample_path, advs[0]):
            output_filename = os.path.join(output_dir, os.path.basename(p))
            image.save_img(output_filename, img.numpy());

        max_pixel_delta = np.max(np.abs(images.numpy() - advs_images.numpy()))
        print("max_pixel_delta", max_pixel_delta)

        # calculate and report the robust accuracy
        """
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        for eps, acc in zip(epsilons, robust_accuracy):
            print(eps, acc.item())

        # we can also manually check this
        for eps, advs_ in zip(epsilons, advs):
            print(eps, accuracy(fmodel, advs_, labels))
            # but then we also need to look at the perturbation sizes
            # and check if they are smaller than eps
            print((advs_ - images).norms.linf(axis=(1, 2, 3)).numpy())
        """


