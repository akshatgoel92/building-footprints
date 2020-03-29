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


def get_image(root = 'Bing Gorakhpur', 
              image_type = 'Bing maps imagery_Gorakhpur', 
              image_name = 'qgis_test.0.tif'):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    
    bucket_name, access_key, secret_access_key = common.\
                                                 get_credentials()
    
    url = 's3://{}/{}/{}/{}'.format(bucket_name, 
                                    root, 
                                    image_type, 
                                    image_name)
    
    session = boto3.Session(aws_access_key_id=access_key, 
                            aws_secret_access_key=secret_access_key)

    with rasterio.Env(AWSSession(session)):
        f = rasterio.open(url)
                
    
    return(f)


def get_masks(root = 'Bing Gorakhpur', 
              image_type = 'Bing maps imagery_Gorakhpur', 
              image_name = 'qgis_test.0.tif', 
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
    
    rstr = get_image(root, image_type, image_name)
    
    with rstr as image:
        
        out_image, out_transform = rasterio.mask.\
                                   mask(image, 
                                        shapes, 
                                        crop = False,
                                        invert = invert,
                                        filled = filled)
        
        out_meta = image.meta
        
    return(out_image, out_transform, out_meta)


def get_labels_from_mask(mask):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    labels = (np.sum(out_image.mask, axis = 0) > 0)\
             .astype(int)\
             .flatten()
    
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
        flat.append(np.array([arr[band][row, col] for row in height for col in width]))
    
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
        file_to = os.path.join(, image_name)
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