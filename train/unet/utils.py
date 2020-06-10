# Import packages
import os
import sys

import time
import json
import keras
import random
import skimage
import train_dl
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


def get_settings(model_type):
    """
    ---------------------------------------------
    Deal with recale argument: need to reciprocate
    Deal with converting lists to tuples
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    path = os.path.join(model_type, "settings.json")

    with open(path) as f:
        settings = json.load(f)

    for config in settings.values():
        config.update({setting: tuple(val) for setting, val in config.items() if type(val) == list})

    return settings


def get_paths(train_frames, train_masks, val_frames, val_masks):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    paths = []

    for folder in [train_frames, train_masks, val_frames, val_masks]:
        paths.append(common.get_local_image_path("data", folder))

    return paths


def check_input_directories(frames, masks):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    frames_to_remove = set(os.listdir(frames)) - set(os.listdir(masks))

    for frame in frames_to_remove:
        os.remove(os.path.join(frames, frame))

    return


def check_folders(paths, extension):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    # Get the filepaths

    # Make sure train and test folders have the same data-sets
    for args in [(paths[0], paths[1]), (paths[2], paths[3])]:
        check_input_directories(*args)

    for folder in paths:
        new_folder = folder.split("/")[1].split("_")[0]
        try:
            os.makedirs(common.get_local_image_path(folder, new_folder))
        except FileExistsError as e:
            print("Directory exists so not making a new one...")
            continue

    for folder in paths:
        for f in os.listdir(folder):
            if f.endswith(extension):
                new = folder.split("/")[1].split("_")[0]
                dest = os.path.join(folder, new)
                source = os.path.join(folder, f)
                os.rename(source, os.path.join(dest, f))


def get_checkpoint_callback(checkpoint_path):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    # Create absolute path to checkpoint
    checkpoint_path = os.path.join("results", checkpoint_path)

    # Add checkpoints for regular saving
    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)

    return checkpoint_cb


def get_early_stopping_callback(patience=10):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    return early_stopping_cb


def get_tensorboard_directory_callback():
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    root_logdir = os.path.join(os.curdir, "logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    callback = keras.callbacks.TensorBoard(os.path.join(root_logdir, run_id))

    return callback
