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
    
    
def train(epochs=2, checkpoint_path = "my_keras_model.h5"):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    # Add checkpoints for regular saving
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True
    )
    
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    # Add TensorBoard logging
    tensorboard_cb = keras.callbacks.TensorBoard(make_tensorboard_directory())

    # Prepare iterators
    train_it, test_it = load_dataset()
    
    if checkpoint_path is not None:
        # Load model:
        model = keras.models.load_model(checkpoint_path)
        # Finding the epoch index from which we are resuming
        initial_epoch = get_init_epoch(checkpoint_path)
    else:
        model = model = unet.define_model()
        initial_epoch = 0

    # Fit model
    history = model.fit_generator(
        train_it,
        steps_per_epoch=675,
        validation_data=test_it,
        validation_steps=225,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
        epochs=epochs,
        verbose=1,
    )
    # Evaluate model
    return(history, model)



    
if __name__ == '__main__':
    
    history, model = train(epochs = 2)
    
    
    summarize_diagnostics(history)