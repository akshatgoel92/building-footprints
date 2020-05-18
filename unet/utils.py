# Import packages
import os
import sys
import unet
import time
import json
import keras
import random
import skimage
import numpy as np


from numpy import load
from keras import backend
from helpers import common
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from skimage import io
from skimage import transform


def get_settings(path):
    """
    ---------------------------------------------
    Deal with recale argument: need to reciprocate
    Deal with converting lists to tuples
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    with open(path) as f:
        settings = json.load(f)
    
    for config in settings.values():
        config.update(
              {
                setting: tuple(val) 
                for setting, val in config.items() 
                if type(val) == list
              }
        )
        
    model_args = settings['model_args']
    output_args = settings['output_args']
    training_args = settings['training_args']
    load_dataset_args = settings['load_dataset_args']
    
    return(model_args, output_args, training_args, load_dataset_args)
    
    
    
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
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True
    )

    return checkpoint_cb


def get_early_stopping_callback():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

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


def iou_coef(y_true, y_pred, smooth=1):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    axes = list(range(1, len(y_true.shape)))
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=axes)

    union = backend.sum(y_true, axes) + backend.sum(y_pred, axes) - intersection
    iou = backend.mean((intersection + smooth) / (union + smooth), axis=0)

    print(iou)

    return iou


def dice_coef(y_true, y_pred, smooth=1):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    axes = list(range(1, len(y_true.shape)))
    intersection = backend.sum(y_true * y_pred, axis=axes)
    union = backend.sum(y_true, axis=axes) + backend.sum(y_pred, axis=axes)
    dice = backend.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    print(dice)

    return dice


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


def create_default_gen(train, mask, mode, 
                       rescale, shear_range, 
                       zoom_range, horizontal_flip, 
                       batch_size, class_mode, target_size, 
                       mask_color, data_format):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    keras.backend.set_image_data_format(data_format)

    gen = ImageDataGenerator(
        rescale=1.0 / rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
    )

    train_gen = (
        img[0]
        for img in gen.flow_from_directory(
            train, 
            batch_size=batch_size, 
            class_mode=class_mode, 
            target_size=target_size
        )
    )

    mask_gen = (
        img[0]
        for img in gen.flow_from_directory(
            mask,
            batch_size=batch_size,
            class_mode=class_mode,
            target_size=target_size,
            color_mode=mask_color,
        )
    )

    gen = (pair for pair in zip(train_gen, mask_gen))

    return gen


def create_custom_gen(train, mask, mode, 
                      rescale, shear_range, 
                      zoom_range, horizontal_flip, 
                      batch_size, class_mode, target_size, 
                      mask_color, data_format):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    # List of training images
    img_type = img_folder.split("/")[1].split("_")[0]
    mask_type = mask_folder.split("/")[1].split("_")[0]

    n = common.list_local_images(img_folder, img_type)
    random.shuffle(n)
    c = 0

    while True:
        
        img = np.zeros((batch_size, 
                        target_size[0], 
                        target_size[1], 
                        channels)).astype("float")
        
        mask = np.zeros((batch_size, 
                         target_size[0], 
                         target_size[1], 1)).astype("float")
        
        for i in range(c, c + batch_size):

            img_path = common.\
                       get_local_image_path(img_folder, img_type, n[i])
            mask_path = common.\
                        get_local_image_path(mask_folder, img_type, n[i])

            train_img = skimage.io.imread(img_path) / rescale
            train_img = skimage.transform.resize(train_img, target_size)
            
            img[i - c] = train_img

            # Need to add extra dimension to mask for channel dimension
            train_mask = skimage.io.imread(mask_path) / rescale
            train_mask = skimage.transform.resize(train_mask, target_size)
            train_mask = train_mask.reshape(target_size[0], target_size[1], 1)
            
            mask[i - c] = train_mask

        # Need to recheck this
        c += batch_size
        if c + batch_size >= len(n):
            c = 0
            random.shuffle(n)

        yield img, mask


def load_dataset(**load_dataset_args):
    """
    ---------------------------------------------
    Input: N/A
    Output: Planet data split into train and test
    ---------------------------------------------
    """
    # Train data generator
    if custom == 1:
        create_gen = create_custom_gen
    else:
        create_gen = create_default_gen

    train = create_gen(**load_dataset_args)
    val = create_gen(**load_dataset_args)

    return (train, val)