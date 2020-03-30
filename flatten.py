import os
from helpers import raster
from helpers import common 

def main():
    
    root = "Bing Gorakhpur"
    image_type = "Bing maps imagery_Gorakhpur"
    shape_root = 'Image Folder'
    shape_type = 'Deoria Metal Shapefile'
    shape_name = 'Metal roof.shp'
    flat_storage = 'flat'
    
    prefix = common.get_s3_paths(image_type, root)
    suffix = ".tif"
    
    files = [img for img in common.get_matching_s3_keys(prefix, suffix)]
    counter = 0
    
    for f in files:
        
        print("This is file number {}".format(counter))
        print(f)
        
        img = raster.get_image(f)
        mask, _, _ = raster.get_masks(img, shape_root,
                                      shape_type, shape_name)
        
        
        labels = raster.get_labels_from_mask(mask)
        f_name = os.path.basename(os.path.splitext(f)[0]) + '.npz'
        
        flat = raster.convert_img_to_flat_file(img, labels)
        raster.write_flat_file(flat, root, flat_storage, f_name)
        
        counter += 1
        
    


if __name__ == '__main__':
    
    main()