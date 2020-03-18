import fiona
import argparse

def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)

def parse_args():
    
    parser = argparse.ArgumentParser(description ='Data loading parser')
    parser.add_argument('--path', type = str, required = False, 
                        default = './Image Folder/Deoria Metal Shapefile/Metal roof.shp')
    
    args.parser.parse_args()
    path = args.path
    
    return(path)


def main():
    
    path = parse_args()
    shp = open_shape_file(path)
    
    return(shp)

if __name__ == '__main__':
    
    shp = main()