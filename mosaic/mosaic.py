import rasterio
import argparse
import os


from rasterio.merge import merge
from rasterio.plot import show
from helpers import common
from helpers import vector
from helpers import raster
from clize import run


def get_image_list(path, extension, chunksize):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    images = [img for img in common.get_matching_s3_keys(path, extension)]
    images = [images[x : x + chunksize] for x in range(0, len(images), chunksize)]

    return images


def open_image_list(images):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    files = []

    for f in images:
        src = raster.open_image(f)
        files.append(src)

    return files


def get_mosaic(files):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    mosaic, out_trans = merge(files)

    for f in files:
        f.close()

    return (mosaic, out_trans)


def write_mosaic(mosaic, out_trans, out_meta, out_fp):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    out_meta.update(
        {"driver": "GTiff", "height": mosaic.shape[1], 
        "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw",}
    )

    with rasterio.open(out_fp, "w", **out_meta, BIGTIFF="IF_NEEDED") as dest:
        dest.write(mosaic)


def main(chunksize=100, 
         extension='.tif', 
         root = "tests", 
         img_type = "chip"):
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
    
    
    path = common.get_local_folder_path(root, img_type)
    
    images = common.list_local_images(root, img_type)
    images = [os.path.join(path, img) for img in images]
    
    for count, element in enumerate(images):
        
        print(count)
        files = open_image_list(element)
        out_meta = files[0].meta.copy()

        mosaic, out_trans = get_mosaic(files)
        out_fp = "chunk_{}.tif".format(count)
        write_mosaic(mosaic, out_trans, out_meta, out_fp)
        
        
if __name__ == "__main__":
    run(main)
