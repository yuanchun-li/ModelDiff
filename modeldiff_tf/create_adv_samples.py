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
    dataset = MyDataset("Imagenet")
    for model_name in pretrain_models:
        print(model_name)
        model = pretrain_models[model_name]
        # model = tf.keras.applications.ResNet50(weights="imagenet")
        # pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
        # bounds = (0, 255)
        pre = dict()
        bounds = (-1, 1)
        fmodel = TensorFlowModel(model, bounds, preprocessing=pre)

        output_dir = "advs_images_tf_%s" % model_name
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        # get data and test the model
        # wrapping the tensors with ep.astensors is optional, but it allows
        # us to work with EagerPy tensors in the following
        # images = ep.astensor(tf.convert_to_tensor(np.vstack(sampled_data)))
        for batch_idx in range(20):
            print("round", batch_idx)
            sampled_data = dataset.sample(50)
            images = ep.astensor(tf.convert_to_tensor(preprocess_input(np.concatenate(sampled_data))))
            labels = ep.astensor(tf.convert_to_tensor(np.stack(random.sample(range(0, 1000), 50))))
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

            sample_path = dataset.get_sample_path()
            max_pixel_delta = 0.
            for pred, advs_pred, p, img, advs_img in zip(predictions, advs_predictions, sample_path, images, advs[0]):
                if pred != advs_pred:
                    output_filename = os.path.join(output_dir, os.path.basename(p))
                    image.save_img(output_filename, advs_img.numpy());
                    max_pixel_delta = max(max_pixel_delta, np.max(np.abs(img.numpy() - advs_img.numpy())))

            print("max_pixel_delta", max_pixel_delta)

