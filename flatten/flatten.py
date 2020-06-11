import os
import platform
import numpy as np
import pandas as pd

from clize import run
from helpers import raster
from helpers import common

# Import oackages
import os
import boto3
import fiona
import argparse
import rasterio
import numpy as np
import rasterio.mask
import matplotlib.pyplot as plt




def convert_img_to_flat_file(img, label):
    """
    --------------------------
    Input:
    Output:
    --------------------------
    """
    img = raster.open_image(img)
    label = raster.open_image(label)
    
    trans = img.transform
    bands = range(img.count)
    width = range(img.width)
    height = range(img.height)
    
    img = raster.convert_img_to_array(img)
    label = raster.convert_img_to_array(label)
    
    flat = []
    geo = [trans * (row, col) for row in height for col in width]
    x = []
    y = []
    for x_, y_ in geo:
        x.append(x_)
        y.append(y_)

    # Get row and columns
    img_coords = [(row, col) for row in height for col in width]
    row = []
    col = []
    
    for element in img_coords:
        row.append(int(element[0]))
        col.append(int(element[1]))
    
    # Put everything together here
    flat.append(np.array(x))
    flat.append(np.array(y))
    flat.append(np.array(row))
    flat.append(np.array(col))
    
    # Now add the raster
    for band in bands:
        flat.append([img[band][row, col] for row in height for col in width])
    
    # Now add the label
    label = list((np.sum(label, axis=0) > 0).astype(int).flatten())
    flat.append(np.array(label))
    
    flat = pd.DataFrame(flat)
    return(flat)
    
    
def write_file(flat, f_name):
    """
    --------------------------
    Input:
    Output:
    --------------------------
    """
    np.savez_compressed(f_name, flat)
    
    
def main(output_format = ".npz", root = "tests", 
         image_type = "train", extension = ".tif", 
         storage = "train", mask = 'mask'):
    """
    Takes as input the tile and returns chips.
    ==========================================
    :width: Desired width of each chip.
    :height: Desired height of each chip.
    :out_path: Desired output file storage folder.
    :in_path: Folder where the input tile is stored.
    :input_filename: Name of the input tile
    :output_filename: Desired output file pattern
    ===========================================
    """
    prefix = common.get_local_folder_path(root, image_type)
    prefix_mask = common.get_local_folder_path(root, mask)
    prefix_storage = common.get_local_folder_path(root, storage)
    args = (output_format, extension, storage, prefix, prefix_storage)
    
    remaining = common.get_remaining(*args)
    masks = common.list_local_images(root, mask)
    
    remaining = [os.path.join(prefix, f) for f in remaining]
    masks = [os.path.join(prefix_mask, f) for f in masks]
    
    counter = 0
    for img, label in zip(remaining, masks[-len(remaining):]):
        
        counter += 1
        print(counter)
        
        try:
            
            f_name = os.path.splitext(os.path.basename(img))[0]
            f_name = os.path.join(prefix_storage, f_name)
            
            flat = convert_img_to_flat_file(img, label)
            write_file(flat, f_name)
        
        except Exception as e:
            print(e)
            continue
            
            
if __name__ == "__main__":
    run(main)