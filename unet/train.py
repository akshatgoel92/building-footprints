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
    path_args,
    training_args,
    model_args,
    output_args,
    load_dataset_args,
    checkpoint_args,
    extension_args,
):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    checkpoint_path = checkpoint_args['checkpoint_path']
    data_format = training_args['data_format']
    
    paths = utils.get_paths(**path_args)
    utils.check_folders(paths, **extension_args)

    # Data format setting
    keras.backend.set_image_data_format(data_format)

    # Callbacks
    callbacks = []
    callbacks.append(utils.get_early_stopping_callback())
    callbacks.append(utils.get_tensorboard_directory_callback())
    callbacks.append(utils.get_checkpoint_callback(checkpoint_path))

    # Prepare iterators
    train_it, test_it = utils.load_dataset(**load_dataset_args)

    # Load model if there are pretrained wets
    if pretrained:
        model = keras.models.load_model(checkpoint_path)
    # Else start a new model
    else:
        model = unet.define_model(**model_args)

    # Fit model
    history = model.fit_generator(train_it, **training_args)

    return (history, model)


def main(model_type="unet"):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    settings = os.path.join(model_type, "settings.json")
    settings = utils.get_settings(settings)
    history, model = train(**settings)
    utils.summarize_training(history)


if __name__ == "__main__":
    main()