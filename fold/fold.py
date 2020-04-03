# Import packages
import os
import math
import boto3
import fiona
import random
import argparse
import rasterio
import numpy as np
import pandas as pd
import rasterio.mask
import matplotlib.pyplot as plt


from helpers import vector
from helpers import common
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.session import AWSSession


def shuffle_flat_files(prefix, suffix):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    random.seed(a=243, version=2)
    
    files = list(common.get_matching_s3_keys(prefix, suffix))
    random.shuffle(files)
    
    chunksize = math.ceil(len(files) / 4)
    chunks = range(0, len(files), chunksize)
    files = [files[x : x + chunksize] for x in chunks]

    return files


def get_flat_file(f):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    flat = np.load(common.get_object_s3(f), allow_pickle = True)['arr_0']
    
    return flat


def merge_flat_file(df1, df2):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = []
    # Assume that the code is has the mask at the end
    for a, b in zip(df1[0:-1], df2[0:-1]):
        df.append(np.concatenate((a, b)))
    
    df.append(np.ma.concatenate((df1[-1], df2[-1])))
    
    return(np.array(df))


def write_block(df, block):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    np.savez_compressed('output_{}.npz'.format(block), df)


def get_block(root, image_type, extension):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    suffix = extension
    prefix = common.get_s3_paths(root, image_type)
    files = shuffle_flat_files(prefix, suffix)
    
    for block, chunk in enumerate(files):
        df = get_flat_file(chunk[0])
        
        for i in range(len(chunk)-1):
          print(i)
          df1 = get_flat_file(chunk[i+1])
          df = merge_flat_file(df, df1)
        
        write_block(df, block)