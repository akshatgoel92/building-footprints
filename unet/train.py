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
    steps_per_epoch=964,
    validation_steps=324,
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
    train_it, test_it = utils.load_dataset(train_frames, 
                                           train_masks, 
                                           val_frames, 
                                           val_masks)

    # Load model if there are pretrained wets
    if pretrained:
        model = keras.models.load_model(checkpoint_path)
    # Else start a new model
    else:
        model = unet.define_model()

    # Fit model
    history = model.fit_generator(
        train_it,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_it,
        validation_steps=validation_steps,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1,
    )
    # Evaluate model
    return (history, model)


def main(settings):
    
    # These are the arguments for downsampling
    conv2d_args = {
        "kernel_size": kernel_size,
        "activation": activation,
        "strides": strides,
        "padding": padding,
        "kernel_initializer": kernel_initializer,
    }
    
    # These are the arguments for upsampling
    conv2d_trans_args = {
        "kernel_size": kernel_size,
        "activation": activation,
        "strides": (2, 2),
        "padding": padding,
        "output_padding": (1, 1),
    }
    
    # These are the arguments for max. pooling
    maxpool2d_args = {
        "pool_size": pool_size,
        "strides": pool_strides,
        "padding": pool_padding,
    }

    paths = utils.get_paths(train_frames, 
                            train_masks, 
                            val_frames, 
                            val_masks)
    
    utils.check_folders(paths)

    history, model = train(*paths, epochs=10)
    utils.summarize_diagnostics(history)


if __name__ == '__main__':
    settings = utils.load_settings()
    main(**settings)