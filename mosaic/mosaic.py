import rasterio
import os

from rasterio.merge import merge
from rasterio.plot import show
from helpers import common 
from helpers import vector
from helpers import raster


def get_image_list(root = 'Bing Gorakhpur', 
                   image_type = 'Bing maps imagery_Gorakhpur', 
                   chunksize = 200):
    
    images = raster.list_images(root, image_type)
    images = [common.get_local_image_path(root, image_type, img) for img in images]
    images = [images[x:x+chunksize] for x in range(0, len(images), chunksize)]
    
    return(images)
    

def open_image_list(images):
    files = []
    i = 1
    for f in images:
        src = rasterio.open(f)
        files.append(src)
        i +=1
    return(files)


def get_mosaic(files):
    
    mosaic, out_trans = merge(files)
    
    for f in files:
        f.close()
    
    return(mosaic, out_trans)


def write_mosaic(mosaic, out_trans, out_meta, count):
    
    out_fp = "./chunk_{}.tif".format(count)
    
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "compress": "lzw"})
    
    with rasterio.open(out_fp, "w", **out_meta, 
                       BIGTIFF="IF_NEEDED") as dest:
        dest.write(mosaic)


if __name__ == '__main__':
    
    images = get_image_list()
    
    for count, element in enumerate(images):
        print(count)
        files = open_image_list(element)
        out_meta = files[0].meta.copy()
        mosaic, out_trans = get_mosaic(files)
        write_mosaic(mosaic, out_trans, out_meta, count)


