import fiona


def open_shape_file(path):
    
    f = fiona.open(path, 'r')
    
    return(f)
    

def tests():
    
    test = open_shape_file('./Image Folder/Deoria Metal Shapefile/Metal roof.shp')
    return(test)

if __name__ == '__main__':
    
    test = tests()