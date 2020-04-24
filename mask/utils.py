# Import oackages
import os
import boto3
import fiona
import argparse
import rasterio
import numpy as np
import rasterio.mask
import matplotlib.pyplot as plt

from helpers import raster
from helpers import vector
from helpers import common
from rasterio.plot import show
from rasterio.session import AWSSession


def get_existing_flat_files(root, image_type):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = get_s3_paths(root, image_type)
    exists = common.get_matching_s3_keys(prefix=prefix, suffix=suffix)

    return exists


def convert_img_to_flat_file(img, labels):
    """''
    --------------------------
    Input:
    Output:
    --------------------------
    """ ""
    arr = raster.convert_img_to_array(img)
    height = range(img.height)
    width = range(img.width)
    bands = range(img.count)
    trans = img.transform
    flat = []

    # Geographic coordinates get stored here
    # Fix a row and iterate through all the columns
    # Then move to the next row
    geo = [trans * (row, col) for row in height for col in width]
    x = []
    y = []
    for x_, y_ in geo:
        x.append(x_)
        y.append(y_)

    # Get row and columns
    # This fixes a row and goes through each column
    # Then it goes to the next row
    img_coords = [(row, col) for row in height for col in width]
    row = []
    col = []
    for row_, col_ in img_coords:
        row.append(row)
        col.append(col)

    # Put everything together here
    flat.append(x)
    flat.append(y)
    flat.append(row)
    flat.append(col)

    # Now add the labels
    for band in bands:
        flat.append(np.array([arr[band][row, col] for row in height for col in width]))

    flat.append(labels)

    return flat


def write_flat_file(
    flat, root="Bing Gorakhpur", image_type="flat", image_name="qgis_test.0.npz"
):
    """
        ------------------------
        Input: 
        Output:
        ------------------------
        """
    file_from = common.get_local_image_path(root, image_type, image_name)
    np.savez_compressed(file_from, flat)

    _, access_key, secret_access_key = common.get_credentials()

    s3_folder = common.get_s3_paths(root, image_type)
    file_to = os.path.join(s3_folder, image_name)
    common.upload_s3(file_from, file_to)