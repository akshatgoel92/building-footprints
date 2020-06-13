import os
import json
import glob

from clize import run
from chip import chip 
from mask import mask 
from split import split 
from mosaic import mosaic
from flatten import flatten 
from train.unet import train
from summarize import summarize 
from predict.unet import predict


def get_settings(path):
    '''
    ====================================================
    Input: Settings file path
    Output: Settings arguments as dictionaries
    ====================================================
    '''
    with open(path) as f:
        settings = json.load(f)
        
    return settings
    
    
def run_chip(chip_args):
    '''
    ====================================================
    Input: Chip module arguments dictionary
    Output: Test results
    ====================================================
    '''
    expected = chip_args.pop('expected')
    chip.main(**chip_args)
    
    
def run_mosaic(mosaic_args): 
     '''
     ====================================================
     Input: Mosaic module arguments dictionary
     Output: Test results
     ====================================================
     '''
     mosaic.main(**mosaic_args)
     
     
def run_mask(mask_args):
    '''
    ====================================================
    Input: 
    Output: 
    ====================================================
    '''
    mask.main(**mask_args)
    
    
def run_flatten(flatten_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    flatten.main(**flatten_args)
    
    
def run_split(split_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    split.main(**split_args)
    
    
def run_summarize(summarize_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    summarize.main(**summarize_args)
    
    
def run_predict(predict_args):
    '''
    ====================================================
    Input: Prediction 
    Output: 
    ====================================================
    '''
    predict.main(**predict_args)
    

def main(prod, chip=False, mask=False, 
         mosaic=False, flatten=False, 
         split=False, summarize=False, predict=False):
    '''
    ====================================================
    Input: 
    Output: 
    ====================================================
    '''
    if mode == "test":
        path = os.path.join("run", "test.json")
        args = get_settings(path)
        
        if chip:
            run_chip(chip_args)
        if mask:
            test_mask(mask_args)
        if mosaic:
            test_mosaic(mosaic_args)
        if flatten:
            test_flatten(flatten_args)
        if split:
            test_split(split_args)
        if summarize:
            test_summarize(sum_args)
        if predict:
            test_predict(predict_args)
        
    elif test == "prod":
        path = os.path.join("run", "prod.json")
        args = get_settings(path)
        
        if chip:
            run_chip(chip_args)
        if mask:
            test_mask(mask_args)
        if mosaic:
            test_mosaic(mosaic_args)
        if flatten:
            test_flatten(flatten_args)
        if split:
            test_split(split_args)
        if summarize:
            test_summarize(sum_args)
        if predict:
            test_predict(predict_args)
        
        
if __name__ == '__main__':
    run(main)