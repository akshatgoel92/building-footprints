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
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = common.get_local_folder_path(root, image_type)

    images = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(".tif")
    ]

    return images


def open_image(image_name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    f = rasterio.open(image_name)
    return(f)


def convert_img_to_array(img):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    return img.read()


def write_image(out_path, out_image, out_meta):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)