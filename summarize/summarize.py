import pandas as pd
import numpy as np

from summarize import utils
from helpers import raster
from helpers import vector


def main():
    
    root = ''
    image_type = ''
    shp_path = ''
    
    files = utils.get_existing_flat_files()
    total_tiles = get_total_tiles(root, image_type)
    
    for img in files:
        
        image = raster.get_image(img)
        arr = raster.convert_img_to_array(image)
        
        res = utils.get_resolution(image)
        area = utils.get_area(image)
        
        avg = utils.get_average(arr)
        sd = utils.get_sd(arr)
        
        # Load shape file
        shp = vector.open_shape_file(shp_path)
        poly = utils.get_poly_area(shp)
        prop = utils.get_area_proportion(shp)
        
        