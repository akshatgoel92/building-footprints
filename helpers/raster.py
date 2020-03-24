# Import oackages
import os
import fiona
import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show
from helpers import common


def list_images(root, image_type):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    path = common.\
           get_local_folder_path(root,
                                 image_type)
    
    images = [f for f in os.listdir(path) 
              if os.path.\
              isfile(os.path.join(path, f))]
    
    return(images)


def get_image(root, image_type, image_name):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    image_path = common.\
                 get_local_image_path(root, 
                                      image_type, 
                                      image_name)
    
    image = rasterio.open(image_path)
    
    return(image)


def get_img_metadata(img):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    return(img.profile)


def convert_img_to_array(img):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    return(img.read())


def write_image(root, image_type, 
                image_name, out_image, out_meta):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    out_path = common.\
               get_local_image_path(root, 
                                    image_type, 
                                    image_name) 
    
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)