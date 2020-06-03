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

from skimage import filters
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

    
    
def get_metadata(img_path):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = raster.open_image(img_path)
    return (img.meta)
     
     
def prep_img(img_path, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    img = skimage.io.imread(img_path)
    img = skimage.transform.resize(img, 
                                  (1, *target_size, channels))
    
    return (img)
    
    
def get_model(model, track):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = keras.models.load_model(model, custom_objects=track)
    return(weights)
    
    
def get_prediction(model, img):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    pred = model.predict(img)[0]
    threshold = filters.threshold_otsu(pred)
    prediction = (pred > threshold).astype('uint8')
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
    
    
def stack_image(img, pred, mask_path, target_size):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    mask_img = prep_img(mask_path, target_size, channels = 1)
    mask_img = mask_img[0]
            
    pred = add_pred_band(pred, img)
    pred = add_mask_band(mask_img, img)
    
    return(pred)
    
    
def write_prediction(dest_path, pred, meta):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    meta['dtype'] = 'uint8'
    meta['width'] = pred.shape[1]
    meta['count'] = pred.shape[-1]
    meta['height'] = pred.shape[0]
    pred = np.moveaxis(pred, -1, 0)
    raster.write_image(dest_path, pred, meta)
    
    
def run_pred(model, track, tests, masks, outputs, target_size, channels, stack):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    weights = get_model(model, track)
    count = 0
    for img_path, mask_path, dest_path in zip(tests, masks, outputs):
        
        count += 1
        print(count)
        test_img = prep_img(img_path, target_size, channels)
        pred = get_prediction(weights, test_img)
        meta = get_metadata(img_path)
        
        test_img = test_img[0]
        write_prediction(dest_path, pred, meta)
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    channels = 8
    img_type = 'val'
    target_size = (640, 640)
    model_name = 'my_keras_model.h5'
    model = os.path.join("results", model_name)
    
    track = {"iou": iou, 
             "dice_coef": dice_coef,
             "iou_thresholded": iou_thresholded}
    
    if img_type == 'val':
        test_outputs_path = os.path.join("data", "val_outputs")
        test_frames_path = os.path.join("data", "val_frames")
        test_masks_path = os.path.join("data", "val_masks")
        
        test_masks = raster.list_images(test_masks_path, img_type)
        test_frames = raster.list_images(test_frames_path, img_type)
    
    elif img_type == 'test':      
        test_outputs_path = os.path.join("data", "test_outputs")
        test_frames_path = os.path.join("data", "test_frames")
        test_masks_path = os.path.join("data", "test_masks")
    
    test_masks = raster.list_images(test_masks_path, img_type)
    test_frames = raster.list_images(test_frames_path, img_type)
    
    tests = [common.get_local_image_path(test_frames_path, img_type, f) for f in test_frames]
    outputs = [common.get_local_image_path(test_outputs_path, img_type, f) for f in test_frames]
    
    masks = [common.get_local_image_path(test_masks_path, img_type, f)  for f in test_masks if f in test_frames]
    results = run_pred(model, track, tests, masks, outputs, target_size, channels, stack = False)
    
    return(results)
    
if __name__ == "__main__":
    results = main()