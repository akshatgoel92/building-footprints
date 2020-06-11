# Import packages
import os
import sys
import time
import json
import train_dl

import random
import skimage
import argparse
import numpy as np
import tensorflow.keras as keras

from clize import run
from skimage import filters
from skimage import img_as_ubyte
from helpers import common
from helpers import raster
from train_dl.metrics import iou
from train_dl.metrics import dice_coef
from train_dl.metrics import jaccard_coef
from train_dl.metrics import iou_thresholded

from numpy import load
from matplotlib import pyplot

from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_metadata(img_path):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = raster.open_image(img_path)
    return img.meta


def prep_img(img_path, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = skimage.io.imread(img_path)
    img = skimage.transform.resize(img, (1, *target_size, channels))

    return img


def get_model(model, track):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = keras.models.load_model(model, custom_objects=track)
    return weights


def get_prediction(model, img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    pred = model.predict(img)[0]
    pred = skimage.filters.gaussian(pred, sigma=2)
    prediction = (pred > pred.mean()).astype("uint8")

    return prediction


def write_prediction(dest_path, pred, meta):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    meta["dtype"] = "uint8"
    meta["width"] = pred.shape[1]
    meta["count"] = pred.shape[-1]
    meta["height"] = pred.shape[0]
    pred = np.moveaxis(pred, -1, 0)
    raster.write_image(dest_path, pred, meta)


def run_pred(model, track, tests, masks, outputs, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    print("Entering prediction loop...")
    weights = get_model(model, track)
    count = 0

    for img_path, mask_path, dest_path in zip(tests, masks, outputs):

        count += 1
        print(count)
        print(dest_path)

        test_img = prep_img(img_path, target_size, channels)
        pred = get_prediction(weights, test_img)
        meta = get_metadata(img_path)

        test_img = test_img[0]
        write_prediction(dest_path, pred, meta)


def main(test=0,
         channels = 8, 
         img_type = "val", 
         model_name = "run_2.h5", 
         target_width = 640,
         target_height = 640,
         test=1):
    """
    Takes as input the a tile and returns chips.
    ==========================================
    :width: Desired width of each chip.
    :height: Desired height of each chip.
    :out_path: Desired output file storage folder.
    :in_path: Folder where the input tile is stored.
    :input_filename: Name of the input tile
    :output_filename: Desired output file pattern
    ===========================================
    """
    target_size = (target_width, target_height)
    
    track = {"iou": iou, "dice_coef": dice_coef, 
            "iou_thresholded": iou_thresholded}
    
    model = os.path.join("results", model_name)
    masks_path = os.path.join("data", "{}_masks".format(img_type))
    frames_path = os.path.join("data", "{}_frames".format(img_type))
    outputs_path = os.path.join("data", "{}_outputs".format(img_type))

    frames = raster.list_images(frames_path, img_type)
    masks = raster.list_images(masks_path, img_type)

    tests = [common.get_local_image_path(frames_path, img_type, f) for f in frames]
    outputs = [common.get_local_image_path(outputs_path, img_type, f) for f in frames]
    masks = [common.get_local_image_path(masks_path, img_type, f) for f in masks if f in frames]

    if test == 1:
        tests = tests[20:30]
        masks = masks[20:30]
        outputs = outputs[20:30]
    
    run_pred(model, track, tests, masks, outputs, target_size, channels)
    
if __name__ == "__main__":
    run(main)