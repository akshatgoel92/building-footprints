# Import packages
import pandas as pd
import numpy as np
import pickle

from helpers import raster
from helpers import vector
from clize import run


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


from helpers import common
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.session import AWSSession
    
    
def get_flat_file(path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = np.load(path, allow_pickle = True)
    df = df['arr_0']
    
    return(df)
    
    
def get_average(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    bands = df[4:-1]
    avgs = np.array([band.mean() for band in bands])

    return avgs


def get_sd(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    bands = df[4:-1]
    sd = np.array([band.std() for band in bands])

    return sd


def get_pixel_count(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    total = len(df[4])
    return total


def get_poly_count(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    mask = df[-1]
    labels = mask.mask.astype(int).sum()

    return labels


def get_poly_proportion(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    mask = df[-1]
    total = len(mask.mask)
    labels = get_poly_count(df)

    return (1 - (labels / total)) * 100


def get_resolution(root, image_type, image_name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = os.path.join(common.get_s3_path(root, image_type), image_name)
    img = raster.get_image(path)
    
    return img.res
    
    
def get_total_tiles(root, image_type):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    return len(raster.list_images(root, image_type))


def get_area(image):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    length = image.bounds.top - image.bounds.bottom
    width = image.bounds.right - image.bounds.left
    area = (length * width) / (1000 * 1000)

    return area


def get_poly_area_shp(shapes):
    """
    ------------------------
    Input: 
    Output:
    Source: https://github.com/scisco/area
    ------------------------
    """
    areas = np.array([area(obj) for obj in shapes]).sum()
    areas = areas / (1000 * 1000)

    return areas


def get_poly_proportion_shp(shapes, path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    vec = vector.get_area(shapes)
    ras = raster.get_area(path)
    proportion = (vec / ras) * 100

    return proportion


def get_summary_df(avgs, sd, bands=3):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df_summary = pd.DataFrame([avgs[0:bands], sd[0:bands]]).T
    df_summary.columns = ["Mean", "SD"]

    return (df_summary)


def get_overlay_data(df, band):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    bands = df[4:-1]
    ones = np.where(df[-1] == 0)
    zeros = np.where(df[-1] == 1)
    df_overlay = [bands[band][ones], bands[band][zeros]]
    
    return df_overlay


def get_regular_data(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    return df[4:-2]


def get_histogram(df, df_summary, dest, f_no=0, overlay=0, x_loc=0.5, y_loc=0.6):
    """
    --------------------------------------------------
    Input: 
    Output:
    --------------------------------------------------
    """
    # Store all histogram arguments here
    hist_args = {
        "histtype": "stepfilled",
        "bins": 500,
        "alpha": 0.6,
        "range": (0, 1000)
    }

    # Store the colors here
    # The max. bands across the imagery we have is six
    # That's why we have chosen six values here
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    # Store annotations here
    title = "Pixel value distribution"
    ylabel = "Normalized probability"
    xlabel = "Value"
    
    # Set up the figure object
    plt.clf()
    ax = plt.gca()
    fig = ax.get_figure()

    # Now iterate through bands and draw each layer
    for i, band in enumerate(df):
        
        # Now prepare image
        band_data = np.unique(band, return_counts=True)
        indices = np.where(band_data[0] > 0)

        # Store the values and weights
        vals = band_data[0][indices]

        # Note that we are normalizing to probabilities
        weights = band_data[1][indices]
        weights = weights / np.sum(weights)
        
        # Make histogram
        ax.hist(vals, weights=weights, label=str(i), color=colors[i], **hist_args)

    # Add annotations
    ax.legend(loc="upper right")
    ax.set_title(title, fontweight="bold")
    
    if overlay:
        name = "hist_masked_{}.png".format(str(f_no))
    else:
        name = "hist_{}.png".format(str(f_no))
        plt.figtext(x_loc, y_loc, df_summary.to_string(), fontsize=8)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(os.path.join(dest, name))
    


def main(bands = 3, suffix = ".npz", root = "tests", image_type = "flat", x_loc = 0.5, y_loc = 0.6):
    """
    Takes as input the a tile and returns chips.
    ==============================================
    :width: Desired width of each chip.
    :height: Desired height of each chip.
    :out_path: Desired output file storage folder.
    :in_path: Folder where the input tile is stored.
    :input_filename: Name of the input tile
    :output_filename: Desired output file pattern
    ===============================================
    """
    files = common.list_local_images(root, image_type, suffix=suffix)
    prefix = common.get_local_folder_path(root, image_type)
    files = [os.path.join(prefix, f) for f in files]
    dest = os.path.join(root, image_type)
    
    for i, f in enumerate(files):

        # Load data
        df = get_flat_file(f)

        # Get information
        avgs = get_average(df)
        sd = get_sd(df)

        # Get summary dataframes
        df_summary = get_summary_df(avgs, sd)

        # Call histograms
        # First make regular histogram
        df_regular = get_regular_data(df)
        get_histogram(df_regular, df_summary, dest, f_no=i, x_loc = x_loc, y_loc = y_loc)
        
        # Now make overlaid histograms
        overlay = []
        
    for j, band in enumerate(range(bands)):
        df_overlay = get_overlay_data(df, band)
        get_histogram(df_overlay, df_summary, dest, f_no=j, overlay=1, x_loc = x_loc, y_loc = y_loc)
            
if __name__ == "__main__":
    run(main)