# Import packages
import os
import time
import keras
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Input,
    concatenate,
)

import sys
from numpy import load
from keras import backend
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import unet


def load_dataset(batch_size=16, target_size=(256, 256)):
    """
    ---------------------------------------------
    Input: N/A
    Output: Planet data split into train and test
    ---------------------------------------------
    """
    # Train data generator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    # Validation dataset
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load dataset
    train_image_generator = train_datagen.flow_from_directory(
        "GE Gorakhpur/data/train_frames/",
        batch_size=batch_size,
        class_mode="input",
        target_size=target_size,
    )
    train_image_generator = (img[0] for img in train_image_generator)

    train_mask_generator = train_datagen.flow_from_directory(
        "GE Gorakhpur/data/train_masks/",
        batch_size=batch_size,
        class_mode="input",
        target_size=target_size,
        color_mode="grayscale",
    )
    train_mask_generator = (img[0] for img in train_mask_generator)

    val_image_generator = val_datagen.flow_from_directory(
        "GE Gorakhpur/data/val_frames/",
        batch_size=batch_size,
        class_mode="input",
        target_size=target_size,
    )
    val_image_generator = (img[0] for img in val_image_generator)

    val_mask_generator = val_datagen.flow_from_directory(
        "GE Gorakhpur/data/val_masks/",
        batch_size=batch_size,
        class_mode="input",
        target_size=target_size,
        color_mode="grayscale",
    )
    val_mask_generator = (img[0] for img in val_mask_generator)

    train_generator = (
        pair for pair in zip(train_image_generator, train_mask_generator)
    )

    val_generator = (pair for pair in zip(val_image_generator, val_mask_generator))

    # Return both forms of data-sets
    return (train_generator, val_generator)


def summarize_diagnostics(history):
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
    pyplot.title("Fbeta")
    pyplot.plot(history.history["fbeta"], color="blue", label="train")
    pyplot.plot(history.history["val_fbeta"], color="orange", label="test")

    # Save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


def make_tensorboard_directory():
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    return os.path.join(root_logdir, run_id)


def train(epochs=2):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    # Add checkpoints for regular saving
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "my_keras_model.h5", save_best_only=True
    )
    
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    # Add TensorBoard logging
    tensorboard_cb = keras.callbacks.TensorBoard(make_tensorboard_directory())

    # Prepare iterators
    train_it, test_it = load_dataset()

    # Define model
    model = unet.define_model()

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
    loss, accuracy = model.evaluate_generator(test_it, steps=225, verbose=1)
    print("> loss=%.3f, accuracy=%.3f" % (loss, accuracy))

    # Learning curves
    summarize_diagnostics(history)
    
    
if __name__ == '__main__':
    train(epochs = 10)
