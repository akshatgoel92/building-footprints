from itertools import product
from rasterio import windows
from helpers import common
from helpers import raster
from helpers import vector

import os
import argparse
import rasterio


def get_tiles(ds, width, height):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    ncols = ds.meta["width"]
    nrows = ds.meta["height"]

    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    for col_off, row_off in offsets:

        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height,
        ).intersection(big_window)

        transform = windows.transform(window, ds.transform)
        yield window, transform


def output_chip(
    in_path, input_filename, out_path, output_filename, width, height,
):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = os.path.join(in_path, input_filename)

    with rasterio.open(path) as inds:
        meta = inds.meta.copy()

        for window, transform in get_tiles(inds, width, height):

            print(window)
            meta["transform"] = transform
            meta["width"], meta["height"] = (
                window.width,
                window.height,
            )

            outpath = os.path.join(
                out_path,
                output_filename.format(int(window.col_off), int(window.row_off)),
            )

            with rasterio.open(outpath, "w", **meta) as outds:
                outds.write(inds.read(window=window))


def upload_chips(in_path, out_path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    upload_files = [
        os.path.join(out_path, img) for img in common.list_local_images(out_path, "")
    ]

    for img in upload_files:
        file_to = common.get_s3_paths(in_path, out_path)
        common.upload_s3(img, os.path.join(in_path, img))


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    in_path = "GE Gorakhpur"
    input_filename = "GC_GOOGLE_V1.tif"

    out_path = "tiles"
    output_filename = "tile_{}-{}.tif"

    width = 256
    height = 256

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--in_path", type=str, default=in_path)
    parser.add_argument("--input_filename", type=str, default=input_filename)

    parser.add_argument("--out_path", type=str, default=out_path)
    parser.add_argument("--output_filename", type=str, default=output_filename,)

    parser.add_argument("--width", type=str, default=width)
    parser.add_argument("--height", type=str, default=height)
    
    args=parser.parse_args()
    common.make_folders(in_path, out_path)
    
    output_chip(**vars(args))
    upload_chips(in_path, out_path)


if __name__ == "__main__":
    main()