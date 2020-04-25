import os
import argparse
import platform
import rasterio
import numpy as np

from helpers import raster
from helpers import vector
from helpers import common


def get_remaining(
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
    files = [img for img in os.listdir(prefix) if img.endswith(extension)
    
    try:

        existing = [
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(prefix_storage) if f.endswith(output_format)
        ]

        remaining = [
            f for f in files if os.path.splitext(os.path.basename(f))[0] not in existing
        ]

    except Exception as e:
        print(e)
        existing = []
        remaining = [f for f in files if os.path.splitext(os.path.basename(f))[0]]

    return remaining


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
    img = raster.get_image(f)

    mask, transform = rasterio.mask.mask(
        img, shapes, crop=False, invert=invert, filled=filled
    )

    mask = (
        (np.sum(mask, axis=0) > 0).astype(int).reshape(1, mask.shape[1], mask.shape[2])
    )

    mask.dtype = "uint8"

    meta = img.meta
    meta["count"] = 1

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


def upload_mask(root, image_type, image_name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    _, access_key, secret_access_key = common.get_credentials()

    s3_folder = common.get_s3_paths(root, image_type)
    file_to = os.path.join(s3_folder, image_name)
    common.upload_s3(file_from, file_to)


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    image_type = os.path.join("data", "val_frames")
    storage = os.path.join("data", "val_masks")
    shape_root = "Metal Shapefile"
    shape_name = "Metal roof.shp"
    shape_type = "Gorakhpur"
    output_format = ".tif"
    root = "GE Gorakhpur"
    extension = ".tif"

    prefix = common.get_local_image_path(root, image_type)
    prefix_storage = common.get_local_image_path(root, storage)

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

    remaining = get_remaining(
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
    )

    shapes = get_shapes(shape_root, shape_type, shape_name)
    counter = 0

    for f in remaining:
        
        counter += 1
        print(f)
        print(counter)
        mask, trans, meta = get_mask(f, shapes)
        f_name = os.path.splitext(os.path.basename(f))[0] + output_format

        try:
            write_mask(mask, meta, root, storage, f_name)
        except Exception as e:
            print(e)
            continue
            
            
if __name__ == "__main__":
    main()