import fiona
import argparse


def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)


