import os
from helpers import raster
from helpers import common
from flatten import utils



def parse_args(root, image_type, shape_root, 
               shape_type, shape_name, output_format, 
               extension, prefix):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    parser = argparse.\
             ArgumentParser(description = '')
    
    parser.add_argument('--root', 
                        type = str, 
                        default = root)
    
    parser.add_argument('--storage', 
                        type = str, 
                        default = storage)
    
    parser.add_argument('--extension', 
                        type = str, 
                        default = extension)
    
    parser.add_argument('--image_type', 
                        type = str, 
                        default = image_type)
    
    parser.add_argument('--shape_root', 
                        type = str, 
                        default = shape_root)
    
    parser.add_argument('--shape_type', 
                        type = str, 
                        default = shape_type)
    
    parser.add_argument('--shape_name', 
                        type = str, 
                        default = shape_name)
    
    parser.add_argument('--output_format', 
                        type = str, 
                        default = output_format)
    
    args = parser.parse_args()
    
    root = args.root
    storage = args.storage
    
    extension = args.extension
    image_type = args.image_type

    shape_root = args.shape_root
    shape_type = args.shape_type
    
    shape_name = args.shape_name
    output_format = args.output_format
    
    prefix = common.\
             get_s3_paths(image_type, root) 

    return(root, image_type, shape_root, shape_type, 
           shape_name, output_format, extension, prefix)

    
def main(root, image_type, shape_root, 
         shape_type, shape_name, output_format, 
         extension, prefix):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''

    files = [img for img in common.\
             get_matching_s3_keys(prefix, extension)]
    
    
    for f in files:
        
        img = raster.get_image(f)
        
        mask, _, _ = utils.\
                     get_masks(img, shape_root, 
                               shape_type, shape_name)
        
        labels = utils.\
                 get_labels_from_mask(mask)
        
        f_name = os.path.basename(os.path.splitext(f)) + \
                 output_format
        
        flat = utils.\
               convert_img_to_flat_file(img, labels)
        
        utils.\
        write_flat_file(flat, root, storage, f_name)



if __name__ == '__main__':
    
    root = "Bing Gorakhpur"
    image_type = "Bing maps imagery_Gorakhpur"
    
    shape_root = 'Image Folder'
    shape_type = 'Deoria Metal Shapefile'
    
    shape_name = 'Metal roof.shp'
    extension = ".tif"
    
    storage = 'flat'
    output_format = '.npz'
    
    args = parse_args(root, image_type, shape_root, shape_type,
                      shape_name, output_format, extension, prefix)
    
    main(*args)