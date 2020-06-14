import os
import json
import glob

from train import unet
from clize import run
from chip import chip 
from mask import mask 
from split import split 
from mosaic import mosaic
from flatten import flatten 
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
    

def main(*, prod=False, chip=False, mask=False, 
         mosaic=False, flatten=False, split=False, 
         summarize=False, predict=False):
    """
    Takes as input the a tile and returns chips.
    ==========================================
    :width: Desired width of each chip.
    :height: Desired height of each chip.
    :out_path: Desired output file storage folder.
    :in_path: Folder where the input tile is stored.
    :input_filename: Name of the input tile
    :output_filename: Desired output file pattern
    ===========================================
    """
    if prod:
        path = os.path.join("run", "prod.json")
        args = get_settings(path)
        
        
    else:
        path = os.path.join("run", "test.json")
        args = get_settings(path)
        
    if chip:
        run_chip(chip_args)
    if mask:
        run_mask(mask_args)
    if mosaic:
        run_mosaic(mosaic_args)
    if flatten:
        run_flatten(flatten_args)
    if split:
        run_split(split_args)
    if summarize:
        run_summarize(sum_args)
    if predict:
        run_predict(predict_args)
        
        
if __name__ == '__main__':
    run(main)