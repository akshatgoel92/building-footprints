# Import packages
import os
import sys
import unet
import time
import json
import keras
import random
import skimage
import numpy as np
import tensorflow as tf

from numpy import load
from keras import backend
from helpers import common
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from skimage import io
from skimage import transform


def create_default_gen(
    train_frame,
    train_mask,
    val_frame,
    val_mask,
    img_type,
    rescale,
    shear_range,
    zoom_range,
    horizontal_flip,
    batch_size,
    class_mode,
    target_size,
    mask_color,
    data_format,
    custom,
    channels,
):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    if img_type = 'train':
        train = train_frame 
        mask = mask_frame
    elif img_type = 'val':
        train = val_frame
        mask = val_mask

    keras.backend.set_image_data_format(data_format)

    gen = ImageDataGenerator(rescale=1.0 / rescale, 
                             shear_range=shear_range, 
                             zoom_range=zoom_range, 
                             horizontal_flip=horizontal_flip,)

    train_gen = (img[0] for img in gen.flow_from_directory(train, 
                 batch_size=batch_size, 
                 class_mode=class_mode, 
                 target_size=target_size))

    mask_gen = (
        img[0] for img in gen.flow_from_directory(mask, 
                                                  batch_size=batch_size, 
                                                  class_mode=class_mode, 
                                                  target_size=target_size, 
                                                  color_mode=mask_color,))

    gen = (pair for pair in zip(train_gen, mask_gen))

    return gen


def get_data_list(img_type, train_frame, train_mask, val_frame, val_mask):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    if img_type == "train":
        frame_root = train_frame
        mask_root = train_mask

    elif img_type == "val":
        frame_root = val_frame
        mask_root = val_mask

    img_list = common.list_local_images(frame_root, img_type)
    random.shuffle(img_list)

    return (img_list, frame_root, mask_root)


def get_data_store(batch_size, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img_store = np.zeros((batch_size, target_size[0], target_size[1], channels)).astype("float")

    return img_store


def prep_mask(mask_path, target_size):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    mask_img = skimage.io.imread(mask_path)
    mask_img = skimage.transform.resize(mask_img, target_size, preserve_range=True)
    mask_img = mask_img.reshape(target_size[0], target_size[1], 1)

    return mask_img


def prep_train(img_path, rescale, target_size):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    train_img = skimage.io.imread(img_path) / rescale
    train_img = skimage.transform.resize(train_img, target_size)

    return train_img


def create_custom_gen(
    train_frame,
    train_mask,
    val_frame,
    val_mask,
    img_type,
    rescale,
    shear_range,
    zoom_range,
    horizontal_flip,
    batch_size,
    class_mode,
    target_size,
    mask_color,
    data_format,
    custom,
    channels,
):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img_list, frame_root, mask_root = get_data_list(img_type, train_frame, train_mask, val_frame, val_mask)
    c = 0

    while True:

        img = get_data_store(batch_size, target_size, channels)
        mask = get_data_store(batch_size, target_size, 1)

        for i in range(c, c + batch_size):

            img_path = common.get_local_image_path(frame_root, img_type, img_list[i])
            mask_path = common.get_local_image_path(mask_root, img_type, img_list[i])

            train_img = prep_train(img_path, rescale, target_size)
            mask_img = prep_mask(mask_path, target_size)

            img[i - c] = train_img
            mask[i - c] = mask_img

        c += batch_size

        if c + batch_size >= len(img_list):
            c = 0
            random.shuffle(img_list)

        yield img, mask


def load_dataset(args1, args2):
    """
    ---------------------------------------------
    Input: N/A
    Output: Planet data split into train and test
    ---------------------------------------------
    """
    if args2["custom"] == 1:
        create_gen = create_custom_gen
    elif args2["custom"] == 0:
        create_gen = create_default_gen

    train = create_gen(*args1, **args2, img_type="train")
    val = create_gen(*args1, **args2, img_type="val")

    return (train, val)
