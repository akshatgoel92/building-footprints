import os
import utils
import argparse
import platform
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

    existing = [
        utils.get_basename(f)
        for f in common.get_matching_s3_keys(prefix_storage, output_format)
    ]

    remaining = [f for f in files if utils.get_basename(f) not in existing]
    counter = 0

    for f in remaining:

        counter += 1

        print(f)
        print(counter)

        try:
            img = raster.get_image(f)
            mask, _, _ = utils.get_masks(img, shape_root, shape_type, shape_name)

            labels = utils.get_labels_from_mask(mask)
            f_name = os.path.splitext(os.path.basename(f))[0] + output_format

            flat = utils.convert_img_to_flat_file(img, labels)
            utils.write_flat_file(flat, root, storage, f_name)

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
    output_format = ".npz"
    root = "GE Gorakhpur"
    image_type = "tiles"
    extension = ".tif"
    storage = "flat"

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
