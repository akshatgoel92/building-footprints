import fiona
from common import helpers


def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)


def get_shapes(root = 'Image Folder', 
               image_type = 'Deoria Metal Shapefile', 
               image_name = 'Metal roof.shp'):
    
    _, path = helpers.get_local_paths(root, 
                                      image_type, 
                                      image_name)
    

    with fiona.open(path, 'r') as shapefile:
        # Need to be careful that no None objects 
        # are in the shapefile
        shapes = [feature["geometry"] 
                  for feature in shapefile 
                  if feature["geometry"] is not None]
    
    return(shapes)