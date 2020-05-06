# Import packages
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import pyproj
import fiona
import json
import os

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
    print(path)
    df = gpd.read_file(path)
    
    return(df)


def merge_geojson(df_list):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = gpd.GeoDataFrame(pd.concat(df_list, ignore_index = True), geometry = 'geometry')
    
    return(df)


def write_geojson_to_shape(df, path, driver = 'GeoJSON'):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df.to_file(path, driver)


def execute_geojson_to_shape(in_path, out_path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df_list = [open_geojson(os.path.join(in_path, vec)) 
               for vec in os.listdir(in_path)]
    
    df = merge_geojson(df_list)
    write_geojson_to_shape(df, out_path)
    
    return


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