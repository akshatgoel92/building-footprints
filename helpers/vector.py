import fiona
import argparse


def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)


def get_shapes(path = './Image Folder/Deoria Metal Shapefile/Metal roof.shp'):
    
    with fiona.open(path, 'r') as shapefile:
        shapes = [feature["geometry"] for feature in shapefile 
                  if feature["geometry"] is not None]
    
    return(shapes)