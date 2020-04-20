# Import packages
import os
import sys
import unet
import time
import keras

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


def predict_model(model = ''):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    if model == '':
        model = keras.models.load_model('my_keras_model.h5')
        iou, dice = get_monitoring_callbacks()
    
    model.summary()
    _, test_it = load_dataset()
    
    y_pred = model.predict_generator(test_it, steps=2, verbose=1, callbacks = [iou, dice])
    return(y_pred)


def evaluate_model(model= None, iou = iou_coef, dice = dice_coef):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    if model is not None:
        model = keras.models.load_model('my_keras_model.h5')
        iou, dice = get_monitoring_callbacks()
    
    model.summary()
    
    _, test_it = load_dataset()
    
    loss, iou, dice, accuracy = model.evaluate_generator(test_it, steps=20, verbose=1, callbacks = [iou, dice])
    print("> loss=%.3f, accuracy=%.3f" % (loss, accuracy))
    
    return(loss, iou, dice, accuracy)


if __name__ == '__main__':
    loss, iou, dice, accuracy = evaluate_model()