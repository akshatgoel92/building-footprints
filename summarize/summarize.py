# Import packages
import pandas as pd
import numpy as np

from summarize import utils
from helpers import raster
from helpers import vector


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


from helpers import vector
from helpers import common
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.session import AWSSession


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


def get_img_metadata(img):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = os.path.join(common.get_s3_path(root, image_type), image_name)
    path = os.path.join(common.get_s3_path(root, image_type), image_name)

    return img.profile


def get_poly_area(path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    shp = vector.open_shape_file(path)


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


def get_summary_df(avgs, sd, area, bands=3):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df_area = pd.DataFrame([area], columns=["Metal roof %"])
    df_summary = pd.DataFrame([avgs[0:bands], sd[0:bands]]).T
    df_summary.columns = ["Mean", "SD"]

    return (df_area, df_summary)


def get_overlay_data(df, band):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    bands = df[4:-2]
    ones = np.where(df[8].data == 0)
    zeros = np.where(df[8].data == 1)
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


def get_histogram(df, f_no=0, overlay=0):
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

        ax.hist(
            vals,
            weights=weights,
            label=str(i),
            color=colors[i],
            range=(0, 260),
            **hist_args
        )

    # Add annotations
    ax.legend(loc="upper right")
    ax.set_title(title, fontweight="bold")

    if not overlay:
        plt.figtext(0.6, 0.5, df_summary.to_string(), fontsize=8)
        plt.figtext(0.6, 0.3, df_area.to_string(), fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    name = "hist_{}.png".format(str(f_no))
    plt.savefig(name)


def main():

    # Convert to command line arguments
    # Need to remove hardcoding from this and other file
    bands = 3
    suffix = ".npz"
    root = "GE Gorakhpur"
    image_type = "blocks"

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--bands", type=int, default=bands)
    parser.add_argument("--suffix", type=str, default=suffix)
    parser.add_argument("--image_type", type=str, default=image_type)

    args = parser.parse_args()
    root = args.root
    bands = args.bands
    suffix = args.suffix
    image_type = args.image_type

    # Get S3 paths
    # List files
    prefix = common.get_s3_paths(root, image_type)
    files = list(common.get_matching_s3_keys(prefix, suffix))

    for i, f in enumerate(files):

        # Load data
        df = utils.get_flat_file(f)

        # Get information
        area = get_poly_area(df)
        avgs = get_avgs(df)
        sd = get_sd(df)

        # Get summary dataframes
        df_area, df_summary = get_dfs(avgs, sd, area)

        # Call histograms
        # First make regular histogram
        df_regular = get_regular_data(df, bands)
        get_histogram(df_regular, f_no=i)

        # Now make overlaid histograms
        overlay = []

        for j, band in enumerate(range(bands)):
            df_overlay = get_overlay_data(df, band)
            get_histogram(df_overlay, f_no=j, overlay=1)


if __name__ == "__main__":
    main()