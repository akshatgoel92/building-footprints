# Import packages
import metrics
import utils
    
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
    pretrained = training_args['pretrained']
    data_format = load_dataset_args['data_format']
    checkpoint_path = checkpoint_args['checkpoint_path']
    
    paths = utils.get_paths(**path_args)
    utils.check_folders(paths, **extension_args)
    keras.backend.set_image_data_format(data_format)
    
    callbacks = []
    callbacks.append(utils.get_early_stopping_callback())
    callbacks.append(utils.get_tensorboard_directory_callback())
    callbacks.append(utils.get_checkpoint_callback(checkpoint_path))
    
    if pretrained:
        model = keras.models.load_model(checkpoint_path)
    else:
        model = unet.define_model(output_args, **model_args)
    
    train, test= utils.load_dataset(paths, 
                                    load_dataset_args)
    
    history = model.fit_generator(train, 
                                  validation_data = test, 
                                  callbacks = callbacks, 
                                  **training_args)
    
    return (history, model)
    
    
if __name__ == "__main__":
    model_type="unet"
    history, model = train(**utils.get_settings(model_type))