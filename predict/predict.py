# Import packages
import os
import sys
import unet
import time
import keras
import numpy as np

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
    

def parse_args(model, track):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    pass

    
def create_test_gen(root, img_type, batch_size, target_size, channels):
    """
    ---------------------------------------------
    Input: N/A
    Output: Tensorboard directory path
    ---------------------------------------------
    """
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
            test_img = skimage.io.imread(img_path) / rescale
            
            test_img = skimage.transform.resize(train_img, target_size)
            img[i - c] = train_img
        
        c += batch_size
        
        if c + batch_size >= len(n):
            c = 0
            random.shuffle(n)

        yield img
    
    
def predict_model(model, track):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """ 
    model = keras.models.load_model(model, custom_objects = track)
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
    model = keras.models.load_model(model, custom_objects =  track)
    _, test_it = load_dataset(track)
    
    results = model.evaluate_generator(test_it, verbose=1)
    return (results)
    
    
def save_model(preds):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    np.savez_compressed("pred.npz", preds)
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    evaluate = False
    predict = True
    
    model = os.path.join("results" ,"my_keras_model.h5")
    track = {"iou": iou, "dice_coef": dice_coef, 
             "iou_thresholded": iou_thresholded,
             "jaccard_coef": jaccard_coef}
    
    if evaluate:
        results = evaluate_model(model, track)
    
    if predict:
        y_pred = predict_model(model, track)
    
    
if __name__ == "__main__":
    results = main()