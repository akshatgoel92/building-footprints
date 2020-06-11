# Import packages
import os
import sys

import time
import json
import tensorflow.keras as keras
import random
import skimage
import numpy as np
import tensorflow as tf

from numpy import load
from train import unet
from helpers import common
from matplotlib import pyplot

from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from skimage import io
from skimage import transform


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    sum_ = backend.sum(K.abs(y_true) + backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def iou(y_true, y_pred, smooth=1.0):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    iou = (intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + smooth)

    return iou


def jaccard_coef(y_true, y_pred):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    intersection = backend.sum(y_true * y_pred)
    union = backend.sum(y_true + y_pred)
    jac = (intersection + 1.0) / (union - intersection + 1.0)
    return backend.mean(jac)


def threshold_binarize(x, threshold=0.5):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.0):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + smooth)


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    coef = (2.0 * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

    return coef


def summarize_training(history):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    # Plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")

    # Plot accuracy
    pyplot.subplot(212)
    pyplot.title("Dice")
    pyplot.plot(history.history["dice_coef"], color="blue", label="train")
    pyplot.plot(history.history["val_dice_coef"], color="orange", label="test")

    pyplot.subplot(213)
    pyplot.title("Intersection over Union")
    pyplot.plot(history.history["iou_coef"], color="blue", label="train")
    pyplot.plot(history.history["val_iou_coef"], color="orange", label="test")

    # Save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()
