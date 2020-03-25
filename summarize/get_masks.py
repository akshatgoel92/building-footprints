from helpers import raster
from helpers import vector 
from helpers import common 

import os
import json
import fiona
import argparse
import rasterio
import rasterio.mask


def get_masks(root, image_type, image_name, 
              shape_root, shape_type, 
              shape_name, invert):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    shapes = vector.\
             get_shapes(shape_root, shape_type, 
                        shape_name)
    
    with raster.\
    get_image(root, image_type, image_name) as image:
        
        out_image, out_transform = rasterio.\
                                   mask.\
                                   mask(image, 
                                        shapes, 
                                        crop = False,
                                        invert = invert)
        
        out_meta = image.meta
    
    return(out_image, out_transform, out_meta)


def get_mask_name(image_name, invert):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    if invert:
        suffix = '_reverse_mask.tif'
    
    else:
        suffix = '_mask.tif'
    
    mask_name = os.path.\
                splitext(image_name)[0] + \
                suffix
    
    return(mask_name)


def get_stack_name(image_name):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    stack_name = os.path.\
                 splitext(image_name)[0] + \
                 '_stacked.tif'
    
    return(stack_name)


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


def stack_masks(root, image_type, stack_name):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    
    names = raster.list_images(root, image_type)
    
    masks = [common.get_local_image_path(root, image_type, name)
             for name in names if 'mask' in name]
    
    with rasterio.open(masks[0]) as f0:
        meta = f0.meta
    
    meta.update(count = len(masks))
    
    with rasterio.open(stack_name, 'w', **meta) as f:
        for id, layer in enumerate(masks, start=1):
            with rasterio.open(layer) as f1:
                f.write_band(id, f1.read(1))
                

def get_args():
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--region', type = str, 
                        default = 'Gorakhpur', required = False)
    
    parser.add_argument('--invert', type = bool, 
                        default = False, required = False)
                            
    args = parser.parse_args()
    region = args.region
    invert = args.invert
    
    with open('summarize/image_config.json', 'r') as f:
        img = json.load(f)[region]
    
    return(img, invert)


def parse_args(img):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    root = img['root']    
    image_type = img['image_type']
    image_name = img['image_name']
    
    shape_root = img['shape_root']
    shape_type = img['shape_type']
    shape_name = img['shape_name']
        
    return(root, image_type, image_name, 
           shape_root, shape_type, shape_name)


def main():
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    img, invert = get_args()
    img = parse_args(img)
    
    out = get_masks(*img, invert)
    root, image_type, image_name = img[0], img[1], img[2]
    
    mask_name = get_mask_name(image_name, invert)
    write_masks(*out, root, image_type, mask_name)
    
    stack_name = get_stack_name(image_name)
    stack_masks(root, image_type, stack_name)

    
if __name__ == '__main__':
    main()