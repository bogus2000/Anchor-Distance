#!/usr/bin/env python
# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

import numpy as np
import random, sys

class kmeans_IoU(object):
    def __init__(self, X, k, max_iter=100):
        self._X = np.array(X)
        self._k = k
        self._max_iter = max_iter

    def _randomSample(self, X, n ):
        sampleix = random.sample(range( X.shape[0] ), int(n) )
        return X[sampleix]

    def _dist_IoU(self, X, centres):
        N, dim = X.shape
        k, cdim = centres.shape
        h_X, w_X = X[:,0], X[:,1]
        X_area = h_X * w_X
        x_min_X, x_max_X, y_min_X, y_max_X = -w_X/2., w_X/2., -h_X/2., h_X/2.
        h_c, w_c = centres[:,0], centres[:,1]
        c_area = h_c * w_c
        x_min_c, x_max_c, y_min_c, y_max_c = -w_c/2., w_c/2., -h_c/2., h_c/2.
        dist = np.zeros((N, k))
        for ik in range(k):
            intersection_x_min = np.max((x_min_X, [x_min_c[ik]] * N), axis=0)
            intersection_y_min = np.max((y_min_X, [y_min_c[ik]] * N), axis=0)
            intersection_x_max = np.min((x_max_X, [x_max_c[ik]] * N), axis=0)
            intersection_y_max = np.min((y_max_X, [y_max_c[ik]] * N), axis=0)
            intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
            IoU = intersection_area / (X_area + c_area[ik] - intersection_area + 1e-9)
            dist[:, ik] = 1. - IoU
        return dist

    def _kmeans(self, X, centres, max_iter=100):
        N, dim = X.shape
        k, cdim = centres.shape
        xtoc, distance = None, None
        for jiter in range(1, max_iter+1):
            D = self._dist_IoU(X=X, centres=centres)
            xtoc = np.argmin(D, axis=1)
            distance = D[np.arange(len(D)), xtoc]
            for jc in range(k):
                c = np.where(xtoc == jc)[0]
                if len(c) > 0:
                    centres[jc] = np.mean(X[c], axis=0)
            sys.stdout.write('{:04d}/{:04d}\r'.format(jiter, max_iter))
        return centres, xtoc, distance

    def kmeansSample(self, nsample=0):
        N, dim = self._X.shape
        if nsample == 0:
            nsample = max(2*np.sqrt(N), 10 * self._k)
        Xsample = self._randomSample(self._X, int(nsample))
        pass1centres = self._randomSample(self._X, int(self._k))
        samplecentres = self._kmeans(Xsample, pass1centres, max_iter=10)[0]
        return self._kmeans(self._X, samplecentres, max_iter=self._max_iter)

# a = np.random.random((1000,2))
# b = kmeans_IoU(X=a, k=3, max_iter=1000)
# centers, xtoc, dist = b.kmeansSample()
# print(centers)
# print(xtoc)
# print(dist)



# import numpy as np
# import time, sys, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # 0 = all messages are logged(default behavior)
# # 1 = INFO messages are not printed
# # 2 = INFO and WARNING messages  arenot printed
# # 3 = INFO, WARNING, and ERROR messages  arenot printed
#
# import tensorflow as tf
#
# # Helper libraries
# import numpy as np
# import os
#
# print(tf.__version__)
#
# a = np.array([[1,2],[1,2,3],[1,2,3,4]])
# b = np.array([[1,1],[1,1],[1,1]])
# print(a.shape)
# strategy = tf.distribute.MirroredStrategy()
# ab = tf.data.Dataset.from_tensor_slices((a, b)).batch(3)
# ab = strategy.experimental_distribute_dataset(ab)

# fashion_mnist = tf.keras.datasets.fashion_mnist
#
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
# # Adding a dimension to the array -> new shape == (28, 28, 1)
# # We are doing this because the first layer in our model is a convolutional
# # layer and it requires a 4D input (batch_size, height, width, channels).
# # batch_size dimension will be added later on.
# train_images = train_images[..., None]
# test_images = test_images[..., None]
#
# # train_images = train_images[0:64*4]
# # train_labels = train_labels[0:64*4]
# print(type(train_labels))
# print(train_images.shape)
#
# # Getting the images in [0, 1] range.
# train_images = train_images / np.float32(255)
# test_images = test_images / np.float32(255)
# print(type(train_labels))
# # If the list of devices is not specified in the
# # `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
# strategy = tf.distribute.MirroredStrategy()
#
# BUFFER_SIZE = len(train_images)
#
# BATCH_SIZE_PER_REPLICA = 64
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
# print(strategy.num_replicas_in_sync)
#
# EPOCHS = 1
#
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
#
# print(type(train_dataset))
# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
#
#
# # print(train_dist_dataset.shape)
#
# def create_model():
#   model = tf.keras.Sequential([
#       tf.keras.layers.Conv2D(32, 3, activation='relu'),
#       tf.keras.layers.MaxPooling2D(),
#       tf.keras.layers.Conv2D(64, 3, activation='relu'),
#       tf.keras.layers.MaxPooling2D(),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(64, activation='relu'),
#       tf.keras.layers.Dense(10)
#     ])
#
#   return model
#
# # Create a checkpoint directory to store the checkpoints.
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#
# with strategy.scope():
#     # Set reduction to `none` so we can do the reduction afterwards and divide by
#     # global batch size.
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#       from_logits=True,
#       reduction=tf.keras.losses.Reduction.NONE)
#     def compute_loss(labels, predictions):
#       per_example_loss = loss_object(labels, predictions)
#       return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
#
# with strategy.scope():
#   test_loss = tf.keras.metrics.Mean(name='test_loss')
#
#   train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#       name='train_accuracy')
#   test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#       name='test_accuracy')
#
# # model, optimizer, and checkpoint must be created under `strategy.scope`.
# with strategy.scope():
#   model = create_model()
#
#   optimizer = tf.keras.optimizers.Adam()
#
#   checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#
# def train_step(inputs):
#   images, labels = inputs
#
#   with tf.GradientTape() as tape:
#     predictions = model(images, training=True)
#     loss = compute_loss(labels, predictions)
#
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#   train_accuracy.update_state(labels, predictions)
#   return loss, loss
#
# def test_step(inputs):
#   images, labels = inputs
#
#   predictions = model(images, training=False)
#   t_loss = loss_object(labels, predictions)
#
#   test_loss.update_state(t_loss)
#   test_accuracy.update_state(labels, predictions)
#
# # `run` replicates the provided computation and runs it
# # with the distributed input.
# @tf.function
# def distributed_train_step(dataset_inputs):
#     per_replica_losses,_ = strategy.run(train_step, args=(dataset_inputs,))
#     # return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
#
# @tf.function
# def distributed_test_step(dataset_inputs):
#   return strategy.run(test_step, args=(dataset_inputs,))
# #
# for epoch in range(EPOCHS):
#   # TRAIN LOOP
#   total_loss = 0.0
#   num_batches = 0
#   for x in train_dist_dataset:
#       # print(len(train_dist_dataset))
#       # print(len(x))
#       # total_loss += distributed_train_step(x)
#       distributed_train_step(x)
#       num_batches += 1
#   # total_loss = distributed_train_step(next(iter(train_dist_dataset)))
#   # print(type(total_loss.numpy()))
#   # num_batches = 1
#   # print(num_batches)
#   train_loss = total_loss / num_batches
#
#   # TEST LOOP
#   for x in test_dist_dataset:
#     distributed_test_step(x)
#
#   if epoch % 2 == 0:
#     checkpoint.save(checkpoint_prefix)
#
#   template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
#               "Test Accuracy: {}")
#   print (template.format(epoch+1, train_loss,
#                          train_accuracy.result()*100, test_loss.result(),
#                          test_accuracy.result()*100))
#
#   test_loss.reset_states()
#   train_accuracy.reset_states()
#   test_accuracy.reset_states()
#
