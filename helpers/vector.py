import fiona
import argparse

def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)

def parse_args():
    
    parser = argparse.ArgumentParser(description ='Data loading parser')
    
    return(path)


def tests():
    
    test = './Image Folder/Deoria Metal Shapefile/Metal roof.shp'
    path = parse_args()
    assert path == test
    
    shp = open_shape_file(path)
    return(shp)

if __name__ == '__main__':
    
    shp = tests()