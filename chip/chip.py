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

        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height,).intersection(big_window)

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

            outpath = os.path.join(out_path, output_filename.format(int(window.col_off), int(window.row_off)),)

            with rasterio.open(outpath, "w", **meta) as outds:
                outds.write(inds.read(window=window))


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    width = 256
    height = 256
    
    out_path = "tiles"
    in_path = "GE Gorakhpur"
    
    input_filename = "GC_GOOGLE_V1.tif"
    output_filename = "tile_{}-{}.tif"
    
    common.make_folders(in_path, out_path)
    output_chip(in_path, input_filename, out_path, output_filename, width, height,)
    

if __name__ == "__main__":
    main()