from itertools import product
from rasterio import windows
from helpers import common
from helpers import raster
from helpers import vector
from clize import run

import os
import argparse
import rasterio


def get_tiles(ds, width, height):
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
    ncols = ds.meta["width"]
    nrows = ds.meta["height"]

    offsets = product(range(0, ncols, width), range(0, nrows, height))
    
    big_window = windows.Window(col_off=0, row_off=0, 
                                width=ncols, height=nrows)

    for col_off, row_off in offsets:

        window = windows.Window(col_off=col_off, row_off=row_off, 
                                width=width, height=height,).intersection(big_window)

        transform = windows.transform(window, ds.transform)
        yield window, transform


def output_chip(in_path, input_filename, out_path, output_filename, width, height,):
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

            outpath = os.path.join(out_path, 
                                   output_filename.\
                                   format(int(window.col_off), 
                                   int(window.row_off)),)

            with rasterio.open(outpath, "w", **meta) as outds:
                outds.write(inds.read(window=window))


def main(width, height, out_path, in_path, 
         input_filename, output_filename):
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
    common.make_folders(in_path, out_path)
    output_chip(in_path, input_filename, out_path, 
                output_filename, width, height,)
    

if __name__ == "__main__":
    run(main)