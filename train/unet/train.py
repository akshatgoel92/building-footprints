# Import packages
import datagen
import metrics
import utils
import unet
import utils


import os
import sys
import time

from numpy import load
from helpers import common
from matplotlib import pyplot

import tensorflow.keras as keras
from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    
def train(
     model_type, path_args, training_args, model_args, 
     output_args, load_dataset_args, checkpoint_args, extension_args,):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    pretrained = training_args.pop("pretrained")
    results_folder = training_args.pop("results_folder")
    
    data_format = load_dataset_args["data_format"]
    checkpoint_path = checkpoint_args["checkpoint_path"]
    
    paths = utils.get_paths(**path_args)
    utils.check_folders(paths, **extension_args)
    keras.backend.set_image_data_format(data_format)
    
    callbacks = []
    callbacks.append(utils.get_early_stopping_callback())
    callbacks.append(utils.get_tensorboard_directory_callback())
    callbacks.append(utils.get_checkpoint_callback(checkpoint_path))
    
    if pretrained:
        model = keras.models.load_model(checkpoint_path)
    
    elif model_type == "unet":
        model = unet.define_model(output_args, **model_args)
    
    train, val = datagen.load_dataset(paths, load_dataset_args)
    history = model.fit_generator(train, validation_data=val, callbacks=callbacks, **training_args)
    
    return (history, model)
    
    
def main():
    
    model_type = "unet"
    history, model = train(model_type, **utils.get_settings(model_type))
    
if __name__ == '__main__':
    main()