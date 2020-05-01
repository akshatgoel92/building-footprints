# Import packages
import geopandas as gpd
import numpy as np
import shapely
import pyproj
import fiona
import json

# Import sub-modules for area calculation
from area import area
from helpers import common
from helpers import raster
from functools import partial
from shapely.ops import transform
from shapely.geometry import shape


def open_geojson(path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = gpd.read_file(path)
    
    return(df)


def merge_geojson(df_list):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = gpd.concat(df_list, axis = 1)
    
    return(df)

def open_shape_file(path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    f = fiona.open(path, "r")

    return f


def get_shapes(shape):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    shapes = [
        feature["geometry"] for feature in shape if feature["geometry"] is not None
    ]

    return shapes


def change_crs(path, target_crs, out_path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    new_df = gpd.read_file(path)
    new_df = new_df.to_crs(target_crs)
    new_df.to_file(out_path)


def write_shape_file(df, path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df.to_file(path)