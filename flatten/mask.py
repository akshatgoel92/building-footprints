import os
import argparse
import platform
import numpy as np

from flatten import utils
from helpers import raster
from helpers import common


def main(
    root,
    image_type,
    shape_root,
    shape_type,
    shape_name,
    output_format,
    extension,
    storage,
    prefix,
    prefix_storage,
):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    print(prefix)
    print(extension)
    files = [img for img in common.get_matching_s3_keys(prefix, extension)]
    
    try: 
        existing = [
            utils.get_basename(f)
            for f in common.get_matching_s3_keys(prefix_storage, output_format)
        ]
        
        
    
    except Exception as e:
        print(e)
        print('This folder does not exist...')
        existing = []
        

    remaining = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in existing]
    counter = 0
    
    for f in remaining:
        counter += 1
        print(f)
        print(counter)

        try:
            img = raster.get_image(f)
            mask, trans, meta = utils.get_masks(img, shape_root, shape_type, shape_name, filled = True)
            
            mask = (np.sum(mask, axis=0) > 0).astype(int).reshape(1, mask.shape[1], mask.shape[2])
            mask.dtype = 'uint8'
            meta['count'] = 1
            
            f_name = os.path.splitext(os.path.basename(f))[0] + output_format
            utils.write_mask(mask, meta, root, storage, f_name)

        except Exception as e:
            print(e)
            continue


def parse_args():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    shape_root = "Metal Shapefile"
    shape_name = "Metal roof.shp"
    shape_type = "Gorakhpur"
    output_format = ".tif"
    root = "GE Gorakhpur"
    image_type = os.path.join('data', 'train_frames')
    extension = ".tif"
    storage = os.path.join('data', 'train_masks')

    prefix = common.get_s3_paths(root, image_type)
    prefix_storage = common.get_s3_paths(root, storage)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--storage", type=str, default=storage)
    parser.add_argument("--extension", type=str, default=extension)
    parser.add_argument("--image_type", type=str, default=image_type)
    parser.add_argument("--shape_root", type=str, default=shape_root)
    parser.add_argument("--shape_type", type=str, default=shape_type)
    parser.add_argument("--shape_name", type=str, default=shape_name)
    parser.add_argument("--output_format", type=str, default=output_format)

    args = parser.parse_args()

    root = args.root
    storage = args.storage
    extension = args.extension
    image_type = args.image_type
    shape_root = args.shape_root
    shape_type = args.shape_type
    shape_name = args.shape_name
    output_format = args.output_format

    return (
        root,
        image_type,
        shape_root,
        shape_type,
        shape_name,
        output_format,
        extension,
        storage,
    )


if __name__ == "__main__":

    args = parse_args()
    print(args)
    main(*args, prefix, prefix_storage)