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
              isfile(os.path.join(path, f))
              and f.endswith('.tif')]
    
    return(images)


def get_image(path):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    
    bucket_name, access_key, secret_access_key = common.\
                                                 get_credentials()
    
    url = 's3://{}/{}'.format(bucket_name, path)
    
    session = boto3.\
              Session(aws_access_key_id=access_key, 
                      aws_secret_access_key=secret_access_key)

    with rasterio.Env(AWSSession(session)):
        f = rasterio.open(url)
                
    
    return(f)

        
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


def write_image(out_path, out_image, out_meta):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)