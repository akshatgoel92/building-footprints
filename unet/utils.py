# Import packages
import os
import sys
import unet
import time
import keras
    
    
from numpy import load
from keras import backend
from helpers import common
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

    
    
def get_paths(train_frames, train_masks, val_frames, val_masks):
    """
    ---------------------------------------------
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    paths =[]
    
    for folder in [train_frames, train_masks, val_frames, val_masks]:
        paths.append(common.get_local_image_path("data", folder))
        
    return(paths)
    
    
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
    
    
def check_folders(paths, extension='.tif'):
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


def create_gen(train,
               mask,
               mode = "train",
               rescale=1.0 / 255, 
               shear_range=0.2, 
               zoom_range=0.2, 
               horizontal_flip=True,
               batch_size=16, 
               class_mode="input", 
               target_size=(256, 256), 
               mask_color = 'grayscale'):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    
    if mode == "train":
        gen = ImageDataGenerator(
            rescale=rescale, 
            shear_range=shear_range, 
            zoom_range=zoom_range, 
            horizontal_flip=horizontal_flip,
    )
    
    elif mode == "validate":
        gen = ImageDataGenerator(rescale=rescale)
                     
    
    train_gen = (img[0] for img in gen.\
                                   flow_from_directory(
                                      train, 
                                      batch_size=batch_size, 
                                      class_mode=class_mode, 
                                      target_size=target_size
                                      ))
    
    mask_gen = (img[0] for img in gen.\
                                  flow_from_directory(
                                      mask, 
                                      batch_size=batch_size, 
                                      class_mode=class_mode, 
                                      target_size=target_size,
                                      color_mode = mask_color,
                                      ))
    
    gen = (pair for pair in zip(train_gen, mask_gen))
    
    return(gen)
    
    
def load_dataset(
                 train_frames,
                 train_masks,
                 val_frames,
                 val_masks,
                 batch_size=16, 
                 target_size=(650, 650), 
                 rescale=1.0 / 255, 
                 shear_range=0.2, 
                 zoom_range=0.2, 
                 horizontal_flip=True,
                 class_mode="input", 
                 mask_color = 'grayscale',
                 ):
    """
    ---------------------------------------------
    Input: N/A
    Output: Planet data split into train and test
    ---------------------------------------------
    """
    # Train data generator
    train = create_gen(train_frames, train_masks)
    val = create_gen(val_frames, val_masks, mode = "validate")
    
    return (train, val)