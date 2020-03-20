from helpers import raster
from helpers import vector 
from helpers import common 

import os
import fiona
import rasterio
import rasterio.mask


def get_masks(root, image_type, image_name, shape):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    shapes = vector.\
             get_shapes(**shape)
    
    with raster.\
    get_image(root, image_type, image_name) as image:
        
        out_image, out_transform = rasterio.\
                                   mask.\
                                   mask(image, 
                                        shapes, 
                                        crop=False)
        
        out_meta = image.meta
    
    return(out_image, 
           out_transform, 
           out_meta)


def write_masks(out_image, out_transform, 
                out_meta, root, image_type, 
                mask_name):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    args = {"driver": "GTiff", 
            "height": out_image.shape[1], 
            "width": out_image.shape[2], 
            "transform": out_transform}
    
    out_meta.update(args)
    raster.write_image(root, 
                       image_type, 
                       mask_name, 
                       out_image, 
                       out_meta)
    
    return(out_image, 
           out_transform, 
           out_meta)


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default = 'Sentinel', required = False)
    parser.add_argument('--image_type', type = str, default= 'Gorakhpur', required = False)
    parser.add_argument('--image_name', type = int, default = 'RT_T44RQQ_20190521T045701_B02.tif', required = False)
    
    parser.add_argument('--shape_root', type = str, default = 'Metal Shapefile', required = False)
    parser.add_argument('--shape_image_type', type = str, default= 'Gorakhpur', required = False)
    parser.add_argument('--shape_image_name', type = int, default = 'Metal roof.shp', required = False)
    args = parser.parse_args()
    
    root = args.root
    image_type = args.image_type
    image_name = args.image_name
    shape_root = args.shape_root
    shape_type = args.shape_type
    shape_name = args.shape_name
    mask_name = os.path.splitext(image_name)[0]
    
if __name__ == '__main__':
    
    out_image, out_transform, out_meta = get_masks()
    write_masks(out_image, out_transform, out_meta)