import os
import argparse
import platform
import rasterio
import numpy as np

from clize import run
from helpers import raster
from helpers import vector
from helpers import common


def get_shapes(shape_root, shape_type, shape_name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    shp = common.get_local_image_path(shape_root, shape_type, shape_name)
    shape = vector.open_shape_file(shp)
    shapes = vector.get_shapes(shape)

    return shapes


def get_mask(f, shapes, invert=False, filled=True):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    img = raster.open_image(f)

    mask, transform = rasterio.mask.mask(img, shapes, crop=False, invert=invert, filled=filled)

    mask = (np.sum(mask, axis=0) > 0).astype(int).reshape(1, mask.shape[1], mask.shape[2])

    mask.dtype = "uint8"

    meta = img.meta
    meta["count"] = 1
    meta["nodata"] = 1
    meta["dtype"] = mask.dtype

    return (mask, transform, meta)


def write_mask(
    mask, meta, root, image_type, image_name,
):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    file_from = common.get_local_image_path(root, image_type, image_name)
    raster.write_image(file_from, mask, meta)


def main(root = "data", image_type = "train_frames_rgb",  shape_root = "data", 
         output_format = ".tif", shape_type = "geojson_buildings", 
         shape_name = "vegas.geojson",  mode = "standard", 
         extension = ".tif", storage = "train_masks_rgb", ):
    """
    Takes as input the a tile and returns chips.
    ==========================================
    :width: Desired width of each chip.
    :height: Desired height of each chip.
    :out_path: Desired output file storage folder.
    :in_path: Folder where the input tile is stored.
    :input_filename: Name of the input tile
    :output_filename: Desired output file pattern
    ===========================================
    """
    prefix = common.get_local_image_path(root, image_type)
    prefix_storage = common.get_local_image_path(root, storage)
    
    if mode == "append":
        out_path = common.get_local_image_path(shape_root, shape_type, shape_name)
        in_path = common.get_local_image_path(shape_root, shape_type)
        vector.execute_geojson_to_shape(in_path, out_path)
        
    shapes = get_shapes(shape_root, shape_type, shape_name)
    
    remaining = [
        common.get_local_image_path(root, image_type, f) for f in common.get_remaining(output_format, 
                                                                                       extension, storage, 
                                                                                       prefix, prefix_storage,)
    ]
    
    counter = 0
    
    for f in remaining:
        
        print(f)
        print(counter)
        
        counter += 1
        mask, trans, meta = get_mask(f, shapes)
        f_name = os.path.splitext(os.path.basename(f))[0] + output_format

        try:
            write_mask(mask, meta, root, storage, f_name)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    run(main)
