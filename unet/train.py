# Import packages
import os
import sys
import unet
import time
import keras

from numpy import load
from keras import backend
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from utils import *

def get_checkpoint_callback():
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
    checkpoint_cb = keras.\
                    callbacks.\
                    ModelCheckpoint(checkpoint_path, 
                                    save_best_only=True)
    
    return(checkpoint_cb)


def get_early_stopping_callback():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    early_stopping_cb = keras.\
                        callbacks.\
                        EarlyStopping(patience=10, restore_best_weights=True)
    
    
    return(early_stopping_cb)
    
    
def train(
    epochs=2,
    pretrained=False,
    results_folder="results",
    checkpoint_path="my_keras_model.h5",
):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    callbacks = []
    
    
    callbacks.append(get_checkpoint_callback())
    callbacks.append(get_early_stopping_callback)
    callbacks.append())
    
    # Prepare iterators
    train_it, test_it = load_dataset()

    if pretrained:
        # Load model:
        model = keras.models.load_model(checkpoint_path)
    else:
        # Get a new model
        model = model = unet.define_model()

    # Fit model
    history = model.fit_generator(
        train_it,
        steps_per_epoch=675,
        validation_data=test_it,
        validation_steps=225,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1,
    )
    # Evaluate model
    return (history, model)


if __name__ == "__main__":

    history, model = train(epochs=5)
    summarize_diagnostics(history)