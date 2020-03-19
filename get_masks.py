from helpers import raster
from helpers import vector 
from helpers import common 

import fiona
import rasterio
import rasterio.mask


def get_masks(root =  'Image Folder', 
              image_type = 'Deoria Landsat 30M', 
              image_name = 'Deoria_2019.tif'):
    
    shapes = vector.get_shapes()
    
    with raster.get_image(root, image_type, image_name) as image:
        out_image, out_transform = rasterio.mask.mask(image, shapes, crop=False)
        out_meta = image.meta
    
    return(out_image, out_transform, out_meta)


def write_masks(out_image, out_transform, 
                out_meta, root = 'Image Folder', 
                image_type = 'Deoria Landsat 30M', 
                mask_name = 'Deoria_2019_mask.tif'):
    
    args = {"driver": "GTiff", 
            "height": out_image.shape[1], 
            "width": out_image.shape[2], 
            "transform": out_transform}
    
    out_meta.update(args)
    raster.write_image(root, image_type, 
                       mask_name, out_image, out_meta)
    
    return(out_image, out_transform, out_meta)

if __name__ == '__main__':
    
    out_image, out_transform, out_meta = get_masks()
    write_masks(out_image, out_transform, out_meta)