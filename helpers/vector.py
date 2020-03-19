import fiona
from helpers import common


def open_shape_file(path):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    f = fiona.open(path, 'r')
    
    return(f)


def get_shapes(root, image_type, image_name):
    '''
    ------------------------
    Input: 
    Output:
    ------------------------
    '''
    path = common.\
           get_local_image_path(root, 
                                image_type, 
                                image_name)
    

    with fiona.open(path, 'r') as shapefile:
        # Need to be careful that no None objects 
        # are in the shapefile
        shapes = [feature["geometry"] 
                  for feature in shapefile 
                  if feature["geometry"] is not None]
    
    return(shapes)