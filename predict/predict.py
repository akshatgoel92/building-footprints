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
from helpers import raster
from unet.metrics import iou
from unet.metrics import dice_coef
from unet.metrics import jaccard_coef
from unet.metrics import iou_thresholded


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
    return (img.meta)
     
     
def prep_test_img(img_path, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    test_img = skimage.io.imread(img_path)
    test_img = skimage.transform.resize(test_img, 
                                        (1, *target_size, channels), 
                                        anti_aliasing = True,
                                        preserve_range = True)
    
    return (test_img)
    
    
def get_model(model, track):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = keras.models.load_model(model, custom_objects=track)
    return(weights)

    
def get_prediction(model, test_img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    pred = model.predict(test_img)[0]
    return (pred)
    
    
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
    
    
def write_prediction(prediction_img, dest_path, meta):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    meta['count'] = prediction_img.shape[-1]
    meta['height'] = prediction_img.shape[0]
    meta['width'] = prediction_img.shape[1]
    meta['dtype'] = str(prediction_img.dtype)
    raster.write_image(dest_path, prediction_img, meta)
    
    
def evaluate_prediction(pred_img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    pass
    
    
def run_pred(model, track, tests, masks, outputs, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = get_model(model, track)
    
    for img_path, mask_path, output_path in zip(tests, masks, outputs):
        
        mask_img = prep_img(mask_path, target_size, channels = 1)
        test_img = prep_img(img_path, target_size, channels)
        pred = get_prediction(weights, test_img)
        
        test_img = test_img[0]
        mask_img = mask_img[0]
        stack_image = add_pred_band(pred, test_img)
        write_prediction(stack_image, output_img, meta)
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    tests = ['/Users/akshatgoel/Desktop/test.tif']
    masks = ['/Users/akshatgoel/Desktop/mask.tif']
    outputs = ['/Users/akshatgoel/Desktop/output.tif']
    
    predict = True
    score = False
    channels = 8

    target_size = (640, 640)
    model_name = 'my_keras_model.h5'
    model = os.path.join("results", model_name)
    
    track = {"iou": iou, 
             "dice_coef": dice_coef, 
             "jaccard_coef": jaccard_coef,
             "iou_thresholded": iou_thresholded}
    
    if predict:
        results = predict_model(model, track, predict_args, steps)
    
    if score
        results = evaluate_model(model, track, predict_args, steps)
    
    return(results)
    
if __name__ == "__main__":
    results = main()