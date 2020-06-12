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
    
    
def test_chip(chip_args):
    '''
    ====================================================
    Input: Chip module arguments dictionary
    Output: Test results
    ====================================================
    '''
    expected = chip_args.pop('expected')
    chip.main(**chip_args)
    
    
def test_mosaic(mosaic_args): 
     '''
     ====================================================
     Input: Mosaic module arguments dictionary
     Output: Test results
     ====================================================
     '''
     mosaic.main(**mosaic_args)
     
     
def test_mask(mask_args):
    '''
    ====================================================
    Input: 
    Output: 
    ====================================================
    '''
    mask.main(**mask_args)
    
    
def test_flatten(flatten_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    flatten.main(**flatten_args)
    
    
def test_split(split_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    split.main(**split_args)
    
    
def test_summarize(summarize_args):
    '''
    ====================================================
    Input: 
    Output:
    ====================================================
    '''
    summarize.main(**summarize_args)
    
    
def test_predict(predict_args):
    '''
    ====================================================
    Input: Prediction 
    Output: 
    ====================================================
    '''
    predict.main(**predict_args)
    
    
def run(chip_args, mosaic_args, 
        mask_args, flatten_args, 
        split_args, sum_args, predict_args): 
    '''
    ====================================================
    Input: Arguments from setting file
    Output: Test results
    ====================================================
    '''
    if "chip" in actions:
        test_chip(chip_args)
    if "mask" in actions:
        test_mask(mask_args)
    if "mosaic" in actions:
        test_mosaic(mosaic_args)
    if "flatten" in actions:
        test_flatten(flatten_args)
    if "split" in actions:
        test_split(split_args)
    if "summarize" in actions:
        test_summarize(sum_args)
    if "predict" in actions:
        test_predict(predict_args)
    
    
def main(mode="test", actions=[""]):
    '''
    ====================================================
    Input: 
    Output: 
    ====================================================
    '''
    if mode == "test":
        path = os.path.join("deploy", "test.json")
        args = get_settings(path)
        run(**args, actions)
        
    elif test == "prod":
        path = os.path.join("deploy", "prod.json")
        args = get_settings(path)
        run(**args, actions)
        
        
if __name__ == '__main__':
    run(main)