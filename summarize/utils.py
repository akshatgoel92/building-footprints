# Import oackages
import os
import boto3
import fiona
import argparse
import rasterio
import numpy as np
import rasterio.mask
import matplotlib.pyplot as plt

from helpers import vector
from helpers import common
from rasterio.plot import show
from rasterio.session import AWSSession


def get_resolution(img):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    return(img.res)


def get_total_tiles(root, image_type):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    return(len(list_images(root, image_type)))


def get_average(arr):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    avgs = [band.mean() for band in arr]

    return(avgs)


def get_sd(arr):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    sd = [band.std() for band in arr]

    return(sd)


def get_area(image):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    length = image.bounds.top - image.bounds.bottom 
    width = image.bounds.right - image.bounds.left
    area = (length*width)/(1000*1000)
    
    return(area)


def get_poly_area(shapes):
    '''
    ------------------------
    Input: 
    Output:
    Source: https://github.com/scisco/area
    ------------------------
    '''
    areas = np.array([area(obj) for obj in shapes]).sum()
    areas = areas/(1000*1000)
    
    return(areas)
    


def calculate_area_proportion(shapes, path):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    vec = vector.get_area(shapes)
    ras = raster.get_area(path)
    proportion = (vec/ras)*100
    
    return(proportion)