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

from helpers import common
from unet.metrics import iou
from unet.metrics import dice_coef
from unet.metrics import jaccard_coef
from unet.metrics import iou_thresholded
from unet.utils import load_dataset

from numpy import load
from keras import backend
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
    
    
def get_settings(model_type="predict"):
    """
    ---------------------------------------------
    Deal with recale argument: need to reciprocate
    Deal with converting lists to tuples
    Input: Keras history project
    Output: Display diagnostic learning curves
    ---------------------------------------------
    """
    path = os.path.join(model_type, "settings.json")
    
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
    
    return (settings)
    
    
def get_metadata(img_path):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = raster.open_image(img_path)
    return (img.transform, img.meta)
     
     
def prep_test_img(img_path):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    test_img = skimage.io.imread(img_path)
    test_img = skimage.transform.resize(test_img, target_size)
    
    return (test_img)
    
    
def get_model(model, track):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    model = keras.models.load_model(model, custom_objects=track)
    return(model)

    
def get_prediction(test_img, model, track, settings):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    prediction = model.predict(test_img)
    return (prediction)
    
    
def add_pred_band(prediction, img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    return(np.dstack((img, prediction)))
    
    
def add_mask_band(mask, img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    return(np.dstack(prediction, img, mask))
    
    
def write_prediction(prediction_img, transform, meta):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = raster.open_img(source_path)
    raster.write_image(dest_path, prediction, meta)
    
    
def load_prediction(prediction_img, transform, meta):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = raster.open_img(source_path)
    return(img)
    
    
def evaluate_prediction(pred_img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    return(img)
    
    
def run_pred(model, track, predict_args, steps):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = get_model(model, track)
    root = os.path.join("data", root)
    test_imgs = common.list_local_images(root, img_type)
    
    for test_img in test_imgs:
        img_path = common.get_local_image_path(root, img_type, test_img)
        transform, meta = get_metadata(img_path)
        test_img = prep_test_img(img_path)
        
        pred = get_prediction(weights, test_img)
        stack_image = add_pred_band(pred, test_img)
        write_prediction(stack_image, dest_path, meta)
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    settings = get_settings()
    steps = settings['misc_args']['steps']
    predict_args = settings['predict_args']
    predict = settings['misc_args']['predict']
    evaluate = settings['misc_args']['evaluate']
    model = os.path.join("results", settings["misc_args"]["model_name"])
    
    track = {"iou": iou, 
             "dice_coef": dice_coef, 
             "jaccard_coef": jaccard_coef,
             "iou_thresholded": iou_thresholded}
    
    if evaluate:
        results = evaluate_model(model, track, predict_args, steps)
    
    if predict:
        results = predict_model(model, track, predict_args, steps)
    
    return(results)
    
if __name__ == "__main__":
    results = main()