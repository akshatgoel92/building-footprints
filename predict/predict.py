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
    
    
def create_test_gen(root, img_type, batch_size, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
    root = os.path.join("data", root)
    n = common.list_local_images(root, img_type)
    
    random.shuffle(n)
    c = 0
    
    while True:
        
        img = np.zeros((batch_size, 
                        target_size[0], 
                        target_size[1], 
                        channels)).astype("float")
                        
                        
        for i in range(c, c + batch_size):
            
            img_path = common.get_local_image_path(root, img_type, n[i])
            test_img = skimage.io.imread(img_path)
            
            test_img = skimage.transform.resize(test_img, target_size)
            img[i - c] = test_img
        
        c += batch_size
        
        if c + batch_size >= len(n):
            c = 0
            random.shuffle(n)

        yield img
    
    
def predict_model(model, track, settings, steps):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """ 
    model = keras.models.load_model(model, custom_objects = track)
    test_it = create_test_gen(**settings)
    
    predictions = model.predict_generator(test_it, steps = steps, verbose=1)
    np.savez_compressed(os.path.join("results", "pred.npz"), predictions)
    
    return predictions
    
    
def evaluate_model(model, track, settings, steps):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    model = keras.models.load_model(model, custom_objects =  track)
    _, test_it = create_test_gen(**settings)
    
    results = model.evaluate_generator(test_it, steps = steps, verbose=1)
    np.savez_compressed(os.path.join("results", "results.npz"), results)
    
    return (results)
    
    
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
    results = os.path.join("results", settings["misc_args"]["model_name"])
    
    track = {"iou": iou, 
             "dice_coef": dice_coef, 
             "jaccard_coef": jaccard_coef,
             "iou_thresholded": iou_thresholded}
    
    if evaluate:
        results = evaluate_model(model, track, predict_args, steps)
    
    if predict:
        y_pred = predict_model(model, track, predict_args, steps)
    
    
if __name__ == "__main__":
    results = main()