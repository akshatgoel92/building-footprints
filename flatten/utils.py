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





def write_mask(
    mask,
    meta,
    root="GE Gorakhpur",
    image_type=os.path.join("data", "train_masks"),
    image_name="test.tif",
):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    file_from = common.get_local_image_path(root, image_type, image_name)
    raster.write_image(file_from, mask, meta)

    _, access_key, secret_access_key = common.get_credentials()

    s3_folder = common.get_s3_paths(root, image_type)
    file_to = os.path.join(s3_folder, image_name)
    common.upload_s3(file_from, file_to)
