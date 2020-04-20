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

import unet


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
    pyplot.title("Dice")
    pyplot.plot(history.history["dice_coef"], color="blue", label="train")
    pyplot.plot(history.history["val_dice_coef"], color="orange", label="test")
    
    pyplot.subplot(213)
    pyplot.title("Intersection over Union")
    pyplot.plot(history.history["iou_coef"], color="blue", label="train")
    pyplot.plot(history.history["val_iou_coef"], color="orange", label="test")

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