import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import sys


# nolbo_multiObjectConfig = {
#     'inputImgDim':[448,448,1],
#     'maxPoolNum':5,
#     'predictorNumPerGrid':11,
#     'bboxDim':5,
#     'class':True, 'zClassDim':64, 'classDim':24,
#     'inst':True, 'zInstDim':64, 'instDim':1000,
#     'rot':True, 'zRotDim':3, 'rotDim':3,
#     'trainable':True,
#     'decoderStructure':{
#         'outputImgDim':[64,64,64,1],
#         'trainable':True,
#         'filterNumList':[512,256,128,64,1],
#         'kernelSizeList':[4,4,4,4,4],
#         'stridesList':[1,2,2,2,2],
#         'activation':tf.nn.leaky_relu,
#         'lastLayerActivation':tf.nn.sigmoid
#     }
# }

def mish(x):
    return x * tf.tanh(tf.keras.activations.softplus(x))


def categorical_crossentropy(gt, pred):
    loss = tf.reduce_mean(-tf.reduce_sum(gt * tf.math.log(pred + 1e-9), reduction_indices=1))
    return loss


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def sampling(mu, logVar):
    epsilon = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    samples = mu + tf.sqrt(tf.exp(logVar)) * epsilon
    return samples


def regulizer_loss(z_mean, z_logVar, dist_in_z_space, class_input=None):
    dim_z = tf.shape(z_mean)[-1]
    batch_size = tf.shape(z_mean)[0]
    z_m_repeat = tf.reshape(z_mean, tf.stack([batch_size, 1, dim_z]))  # shape = (batchSize, 1, dimz)
    z_m_repeat = tf.tile(z_m_repeat, tf.stack([1, batch_size, 1]))  # shape = (batchSize, batchSize, dimz)
    z_m_repeat_tr = tf.transpose(z_m_repeat, [1, 0, 2])
    z_logVar_repeat = tf.reshape(z_logVar, tf.stack([batch_size, 1, dim_z]))
    z_logVar_repeat = tf.tile(z_logVar_repeat, tf.stack([1, batch_size, 1]))

    diff = tf.abs(z_m_repeat - z_m_repeat_tr) / tf.exp(0.5 * z_logVar_repeat)  # shape = (batchSize, batchSize, dimz)
    diff = tf.reduce_sum(diff, axis=-1)  # shape = (batchSize, batchSize)

    diff_in_z = diff - dist_in_z_space * tf.ones_like(diff)
    diff_in_z = tf.where(
        tf.greater(diff_in_z, tf.zeros_like(diff_in_z)), tf.zeros_like(diff_in_z), tf.square(diff_in_z))
    # diff_in_z = tf.reduce_sum(diff_in_z, axis=-1) # shape = (batchSize, )

    if class_input != None:
        c_i_repeat = tf.reshape(class_input,
                                tf.stack([batch_size, 1, tf.shape(class_input)[-1]]))  # shape = (batchSize,1,cdim)
        c_i_repeat = tf.tile(c_i_repeat, tf.stack([1, batch_size, 1]))  # shape = (batchSize, batchSize, cdim)
        c_i_repeat_tr = tf.transpose(c_i_repeat, [1, 0, 2])
        c_i_diff_abs = tf.abs(c_i_repeat - c_i_repeat_tr)  # shape = (batchSize ,batchSize, cdim)
        c_i_diff_sum = tf.reduce_sum(c_i_diff_abs, axis=-1)  # shape = (batchSize, batchSize)
        # if categories are the same, get 1
        # else, get zero
        c_i_diff = tf.where(
            tf.greater(c_i_diff_sum, tf.zeros_like(c_i_diff_sum)), tf.zeros_like(c_i_diff_sum),
            tf.ones_like(c_i_diff_sum))
        diff_in_z = diff_in_z * c_i_diff  # shape = (batchSize, batchSize) = (batchSize,batchSize)*(batchSize,batchSize)

    loss_reg = tf.reduce_sum(diff_in_z, axis=-1)  # shape = (batchSize,)
    # loss_reg = tf.reduce_mean(loss_reg)
    return loss_reg


def binary_loss(xPred, xTarget, epsilon=1e-7, gamma=0.5, b_range=False):
    b_range = float(b_range)
    voxelDimTotal = np.prod(xPred.get_shape().as_list()[1:])
    xTarget = tf.reshape(xTarget, (-1, voxelDimTotal))
    xPred = tf.reshape(xPred, (-1, voxelDimTotal))
    yTarget = -b_range + (2.0 * b_range + 1.0) * xTarget
    yPred = tf.clip_by_value(xPred, clip_value_min=epsilon, clip_value_max=1.0 - epsilon)
    bce_loss = - tf.reduce_sum(
        gamma * yTarget * tf.math.log(yPred) + (1.0 - gamma) * (1.0 - yTarget) * tf.math.log(1.0 - yPred),
        axis=-1)
    return bce_loss


tf.math.log


def kl_loss(mean, logVar, mean_target, logVar_target):
    m, lV = mean, logVar
    m_t, lV_t = mean_target, logVar_target
    # vectorDimTotal = np.prod(mean.get_shape().as_list()[1:])
    # m = tf.reshape(mean, (-1, vectorDimTotal))
    # # m = tf.clip_by_value(m, clip_value_min=-1e+2, clip_value_max=1e+2)
    # lV = tf.reshape(logVar, (-1, vectorDimTotal))
    # # lV = tf.clip_by_value(lV, clip_value_min=-1e+1, clip_value_max=1e+1)
    # m_t = tf.reshape(mean_target, (-1, vectorDimTotal))
    # # m_t = tf.clip_by_value(m_t, clip_value_min=-1e+2, clip_value_max=1e+2)
    # lV_t = tf.reshape(logVar_target, (-1, vectorDimTotal))
    # # lV_t = tf.clip_by_value(lV_t, clip_value_min=-1e+1, clip_value_max=1e+1)
    loss = tf.reduce_sum(0.5 * (lV_t - lV) + (tf.exp(lV) + tf.square(m - m_t)) / (2.0 * tf.exp(lV_t)) - 0.5,
                         axis=-1)
    return loss


def voxelPrecisionRecall(xTarget, xPred, prob=0.5):
    '''
    :param xTarget: voxel
    :param xPred: voxel
    :param prob: threshold
    :return: TP, FP, FN
    {p = TP / (TP + FP + 1e-10),
     r = TP / (TP + FN + 1e-10)}
    '''
    yTarget = tf.cast(tf.reshape(xTarget, (-1, np.prod(xTarget.get_shape().as_list()[1:]))), tf.float32)
    yPred = tf.cast(tf.greater_equal(tf.reshape(xPred, (-1, np.prod(xPred.get_shape().as_list()[1:]))), prob),
                    tf.float32)
    TP = (tf.reduce_sum(yTarget * yPred, axis=-1))
    FP = (tf.reduce_sum((1.0 - yTarget) * yPred, axis=-1))
    FN = (tf.reduce_sum(yTarget * (1.0 - yPred), axis=-1))

    return TP, FP, FN


def nonMaximumSuppresion(boxes, IOUThreshold):
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    # boxes = [box1, ..., boxn]
    # box = [0:colMin, 1:rowMin, 2:colMax, 3:rowMax, 4:objness]
    pickedBoxIdx = []
    cMins = boxes[:, 0]
    rMins = boxes[:, 1]
    cMaxs = boxes[:, 2]
    rMaxs = boxes[:, 3]
    objness = boxes[:, 4]
    area = (rMaxs - rMins) * (cMaxs - cMins) * 1.0
    boxIdxs = np.argsort(objness)
    # print boxIdxs.shape
    while len(boxIdxs) > 0:
        sortedBoxLastIdx = len(boxIdxs) - 1
        boxIdxCurr = boxIdxs[sortedBoxLastIdx]
        pickedBoxIdx.append(boxIdxCurr)
        # get intersection box rowcol
        rrMin = np.maximum(rMins[boxIdxCurr], rMins[boxIdxs[:sortedBoxLastIdx]])
        ccMin = np.maximum(cMins[boxIdxCurr], cMins[boxIdxs[:sortedBoxLastIdx]])
        rrMax = np.minimum(rMaxs[boxIdxCurr], rMaxs[boxIdxs[:sortedBoxLastIdx]])
        ccMax = np.minimum(cMaxs[boxIdxCurr], cMaxs[boxIdxs[:sortedBoxLastIdx]])

        w = np.maximum(0.0, ccMax - ccMin)
        h = np.maximum(0.0, rrMax - rrMin)
        intersection = (w * h)
        union = area[boxIdxCurr] + area[boxIdxs[:sortedBoxLastIdx]] - intersection
        IOU = intersection / union
        # print IOU

        cx_curr = (cMins[boxIdxCurr] + cMaxs[boxIdxCurr]) / 2.
        cy_curr = (rMins[boxIdxCurr] + rMins[boxIdxCurr]) / 2.
        cxs = (cMins[boxIdxs[:sortedBoxLastIdx]] + cMaxs[boxIdxs[:sortedBoxLastIdx]]) / 2.
        cys = (rMins[boxIdxs[:sortedBoxLastIdx]] + rMins[boxIdxs[:sortedBoxLastIdx]]) / 2.
        c_diff2 = np.square(cx_curr - cxs) + np.square(cy_curr - cys)

        r_min_outer = np.minimum(rMins[boxIdxCurr], rMins[boxIdxs[:sortedBoxLastIdx]])
        c_min_outer = np.minimum(cMins[boxIdxCurr], cMins[boxIdxs[:sortedBoxLastIdx]])
        r_max_outer = np.maximum(rMaxs[boxIdxCurr], rMaxs[boxIdxs[:sortedBoxLastIdx]])
        c_max_outer = np.maximum(cMaxs[boxIdxCurr], cMaxs[boxIdxs[:sortedBoxLastIdx]])
        c2 = np.square(c_max_outer - c_min_outer) + np.square(r_max_outer - r_min_outer)

        RDIOU = c_diff2 / c2

        metric = IOU - 0.1 * RDIOU
        # metric = IOU
        # print(RDIOU)

        boxIdxs = np.delete(boxIdxs, np.concatenate(([sortedBoxLastIdx], np.where(metric > IOUThreshold)[0])))
        # print boxIdxs.shape
    return pickedBoxIdx


import numpy as np
import random, sys


class kmeans_IoU(object):
    def __init__(self, X, k, max_iter=100):
        self._X = np.array(X)
        self._k = k
        self._max_iter = max_iter

    def _randomSample(self, X, n):
        sampleix = random.sample(range(X.shape[0]), int(n))
        return X[sampleix]

    def _dist_IoU(self, X, centres):
        N, dim = X.shape
        k, cdim = centres.shape
        h_X, w_X = X[:, 0], X[:, 1]
        X_area = h_X * w_X
        x_min_X, x_max_X, y_min_X, y_max_X = -w_X / 2., w_X / 2., -h_X / 2., h_X / 2.
        h_c, w_c = centres[:, 0], centres[:, 1]
        c_area = h_c * w_c
        x_min_c, x_max_c, y_min_c, y_max_c = -w_c / 2., w_c / 2., -h_c / 2., h_c / 2.
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
        for jiter in range(1, max_iter + 1):
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
            nsample = max(2 * np.sqrt(N), 10 * self._k)
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

class kmeans_dist(object):
    def __init__(self, X):
        self._X = np.array(X)

    def _randomSample(self, X, n):
        sampleix = random.sample(range(X.shape[0]), int(n))
        return X[sampleix]

    def _dist(self, X, centres, log_scale=False, sqr_scale=False):
        N = len(X)
        k = len(centres)
        if log_scale:
            X_ = np.log(X+1)
            centres_ = np.log(centres+1.)
        elif sqr_scale:
            X_ = np.square(X)
            centres_ = np.square(centres)
        else:
            X_ = X
            centres_ = centres
        dist = np.zeros((N, k))
        for ik in range(k):
            d = np.abs(X_ - centres_[ik])
            dist[:, ik] = d
        return dist

    def _kmeans(self, X, centres, max_iter=100, log_scale=False, sqr_scale=False):
        N = len(X)
        k = len(centres)
        xtoc, distance = None, None
        for jiter in range(1, max_iter + 1):
            D = self._dist(X=X, centres=centres, log_scale=log_scale, sqr_scale=sqr_scale)
            xtoc = np.argmin(D, axis=1)
            distance = D[np.arange(len(D)), xtoc]
            for jc in range(k):
                c = np.where(xtoc == jc)[0]
                if len(c) > 0:
                    centres[jc] = np.sqrt(np.mean(np.square(X[c])))
            sys.stdout.write('{:04d}/{:04d}\r'.format(jiter, max_iter))
        return centres, xtoc, distance

    def kmeansSample(self, nsample=0, max_iter=1000, k=10, log_scale=False, sqr_scale=False):
        N = len(self._X)
        if nsample == 0:
            nsample = max(2 * np.sqrt(N), 10 * k)
        Xsample = self._randomSample(self._X, int(nsample))
        pass1centres = self._randomSample(self._X, int(k))
        samplecentres = self._kmeans(Xsample, pass1centres, max_iter=10)[0]
        return self._kmeans(self._X, samplecentres, max_iter=max_iter, log_scale=log_scale, sqr_scale=sqr_scale)
