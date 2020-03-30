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



def get_masks(rstr, 
              shape_root = 'Image Folder', 
              shape_type = 'Deoria Metal Shapefile', 
              shape_name = 'Metal roof.shp', 
              invert = False,
              filled = False):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    shp = common.get_local_image_path(shape_root, 
                                      shape_type, 
                                      shape_name)
    shape = vector.open_shape_file(shp)
    shapes = vector.get_shapes(shape)
    
    out_image, out_transform = rasterio.mask.\
                                        mask(rstr, 
                                             shapes, 
                                             crop = False,
                                             invert = invert,
                                             filled = filled)
        
    out_meta = rstr.meta
        
    return(out_image, out_transform, out_meta)


def get_labels_from_mask(mask):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    labels = (np.sum(mask, axis = 0) > 0)\
             .astype(int).flatten()
    
    return(labels)


def convert_img_to_flat_file(img, labels):
    '''''
    --------------------------
    Input:
    Output:
    --------------------------
    '''''
    arr = convert_img_to_array(img)
    height = range(img.height)
    width = range(img.width)
    bands = range(img.count)
    trans = img.transform
    flat = []
    
    geo = [trans*(row, col) for row in height for col in width]
    x = []
    y = []
    for x_, y_ in geo:
        x.append(x_)
        y.append(y_)
    
    x = np.array(x, dtype = np.float64)
    y = np.array(x, dtype = np.float64)
        
    row = np.array(list(height))
    col = np.array(list(width))
    
    flat.append(x)
    flat.append(y)
    flat.append(row)
    flat.append(col)
    
    for band in bands:
        flat.append(np.array([arr[band][row, col] 
                    for row in height for col in width]))
    
    flat.append(labels)
    
    return(flat)


def write_flat_file(flat,
                    root = 'Bing Gorakhpur', 
                    image_type = 'flat', 
                    image_name = 'qgis_test.0.npz'):
        '''
        ------------------------
        Input: 
        Output:
        ------------------------
        '''
        file_from = common.get_local_image_path(root, 
                                                image_type, 
                                                image_name)
        np.savez_compressed(file_from, flat)
        
        _, access_key, secret_access_key = common.\
                                           get_credentials()
        
        s3_folder = common.get_s3_paths(image_type, root)[0]
        file_to = os.path.join(s3_folder, image_name)
        common.upload_s3(file_from, file_to)
    
    

def load_flat_file(flat, path):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    flat = np.load(path)
    
    return(flat)
    