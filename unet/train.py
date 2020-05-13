# Import packages
from unet import utils
import unet

import os
import sys
import unet
import time
import keras


from numpy import load
from keras import backend
from helpers import common
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD



def train(
    train_frames,
    train_masks,
    val_frames,
    val_masks,
    epochs=2,
    pretrained=False,
    results_folder="results",
    checkpoint_path="my_keras_model.h5",
    data_format="channels_last",
):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    # Data format setting
    keras.backend.set_image_data_format(data_format)

    # Callbacks
    callbacks = []
    callbacks.append(utils.get_early_stopping_callback())
    callbacks.append(utils.get_tensorboard_directory_callback())
    callbacks.append(utils.get_checkpoint_callback(checkpoint_path))

    # Prepare iterators
    train_it, test_it = utils.load_dataset(train_frames, train_masks, val_frames, val_masks)

    # Load model if there are pretrained wets
    if pretrained:
        model = keras.models.load_model(checkpoint_path)
    # Else start a new model
    else:
        model = unet.define_model()

    # Fit model
    history = model.fit_generator(
        train_it,
        steps_per_epoch=964,
        validation_data=test_it,
        validation_steps=324,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1,
    )
    # Evaluate model
    return (history, model)


if __name__ == "__main__":

    train_frames = "train_frames"
    train_masks = "train_masks"
    val_frames = "val_frames"
    val_masks = "val_masks"

    paths = utils.get_paths(train_frames, train_masks, val_frames, val_masks)
    utils.check_folders(paths)

    history, model = train(*paths, epochs=10)
    utils.summarize_diagnostics(history)