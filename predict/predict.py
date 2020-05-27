# Import packages
import os
import sys
import unet
import time
import keras
import numpy as np

from unet import iou_coef
from unet import dice_coef
from utils import load_dataset

from numpy import load
from keras import backend
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
    
    
def parse_args(model):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    pass
    
def predict_model(model, track):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """ 
    model = keras.models.load_model(model, custom_objects=track)
    _, test_it = load_dataset()
    
    predictions = model.predict_generator(test_it, verbose=1)
    return predictions
    
    
def evaluate_model(model, track):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    model = keras.models.load_model(model, custom_objects=track)
    _, test_it = load_dataset(track)
    
    results = model.evaluate_generator(test_it, verbose=1)
    return (results)
    
    
def save_model(preds, results):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    np.savez_compressed("pred.npz", preds)
    np.savez_compressed("results.npz", results)
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    model = "my_keras_model.h5"
    track = {"iou_coef": iou_coef, "dice_coef": dice_coef}
    
    results = evaluate_model(model, track)
    y_pred = predict_model(model, track)
    
    return (y_pred, results)


if __name__ == "__main__":
    results = main()