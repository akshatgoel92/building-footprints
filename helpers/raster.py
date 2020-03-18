# Import oackages
import os
import fiona
import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


def list_images(root = 'Image Folder', 
                image_type = 'Deoria Landsat 30M'):
    '''
    Input: 
    Output: 
    '''
    path = './' + root + '/' + image_type + '/'
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    return(images)


def get_image(root = 'Image Folder', image_type = 'Deoria Landsat 30M', 
              image_name = 'Deoria_2019.tif'):
    '''
    Input: Root folder, image type, image name
    Output: Image reference
    '''
    # Construct the image path
    image_path = './' + root + '/' + image_type + '/' + image_name
    # Open the image
    img = rasterio.open(image_path)
    # Return the image
    return(img)


def get_img_metadata(img):
    '''
    Input: Image reference
    Output: Image metadata dictionary
    '''
    return(img.profile)


def convert_img_to_array(img):
    '''
    Input: Image reference
    Output: Numpy array containing pixel values
    '''
    
    # Read all raster values
    arr = img.read()
    
    return(arr)


def plot_image(img, title = '', y_label = '', band = 1):
    '''
    Inputs: 1) Image reference
            2) Title
            3) Label
    Output: Plot
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    dsmplot = ax.imshow(img.read(band))
    
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(dsmplot, fraction=0.035, pad=0.01)
    
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(y_label, rotation=270)
    
    ax.set_axis_off()
    plt.show()