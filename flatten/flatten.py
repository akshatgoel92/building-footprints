import os
import utils
import argparse
import platform
import numpy as np
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

from helpers import raster
from helpers import vector
from helpers import common
from rasterio.plot import show
from rasterio.session import AWSSession


def convert_img_to_flat_file(img, mask):
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
    
    labels = (np.sum(mask, axis=0) > 0).astype(int).flatten()
    flat.append(labels)

    return flat


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    output_format = ".npz"
    root = "GE Gorakhpur"
    image_type = "tiles"
    extension = ".tif"
    storage = "flat"
    mask = 'train'
    
    prefix = common.get_s3_paths(root, image_type)
    prefix_storage = common.get_s3_paths(root, storage)
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--mask", type=str, default=mask)
    parser.add_argument("--storage", type=str, default=storage)
    parser.add_argument("--extension", type=str, default=extension)
    parser.add_argument("--image_type", type=str, default=image_type)
    parser.add_argument("--output_format", type=str, default=output_format)
    

    args = parser.parse_args()

    
    root = args.root
    mask = args.mask
    storage = args.storage
    extension = args.extension
    image_type = args.image_type
    shape_root = args.shape_root
    shape_type = args.shape_type
    shape_name = args.shape_name
    output_format = args.output_format
    
    
    remaining = common.get_remaining(
        output_format,
        extension,
        storage,
        prefix,
        prefix_storage,
    )
    
    mask = os.path.join(os.path.join('data', '{}_masks'.format(mask)), mask)
    masks = common.list_local_images(root, mask)
    counter = 0

    for rast, mask in zip(remaining, masks[-len(remaining):]:
        
        counter += 1
        print(f)
        print(counter)
        
        try:
            img = raster.get_image(rast)
            mask = raster.get_image(mask)
            flat = convert_img_to_flat_file(img, mask)
            
            os.path.splitext(os.path.basename(f))[0] + output_format
            common.write_flat_file(flat, root, storage, f_name)

        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()