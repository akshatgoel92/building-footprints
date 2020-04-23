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
    
    
    
def get_monitoring_callbacks():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    iou = keras.callbacks.callbacks.LambdaCallback(iou_coef)
    dice = keras.callbacks.callbacks.LambdaCallback(dice_coef)
    
    return(iou, dice)


def predict_model(model = 'my_keras_model.h5'):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    model = keras.models.load_model(model, 
                                    custom_objects = {'iou_coef': iou_coef, 
                                                      'dice_coef': dice_coef})
    model.summary()
    _, test_it = load_dataset()
    
    y_pred = model.predict_generator(test_it, steps=225, verbose=1)
    np.savez_compressed('y_pred.npz', y_pred)
    
    print(y_pred.shape)
    print(y_pred)
    
    return(y_pred)


def evaluate_model(model= 'my_keras_model.h5'):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    model = keras.models.load_model(model, 
                                    custom_objects = {'iou_coef': iou_coef, 
                                                      'dice_coef': dice_coef})
    model.summary()
    
    _, test_it = load_dataset()
    
    loss, accuracy, iou, dice = model.evaluate_generator(test_it, steps=225, verbose=1)
    print("> loss=%.3f, > accuracy=%.3f, > iou=%.3f, dice=%.3f" % (loss, accuracy, iou, dice))
    
    return(loss, iou, dice, accuracy)


def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    loss, iou, dice, accuracy = evaluate_model('my_keras_model.h5')
    y_pred = predict_model()
    
    return(y_pred, loss, iou, dice, accuracy)

if __name__ == '__main__':
    y_pred, loss, iou, dice, accuracy = main()