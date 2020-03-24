from rasterio.plot import show_hist
from matplotlib import pyplot
from helpers import raster
from helpers import common 


import os
import json
import argparse
import rasterio


def get_mask_list():
    '''
    --------------------------------------------------
    Input: 
    Output:
    --------------------------------------------------
    '''
    pass


def plot_histogram(root, img_type, img_name, rvs_name):
     '''
     --------------------------------------------------
     Input: 
     Output:
     --------------------------------------------------
     '''
     img = raster.get_image(root, img_type, img_name)
     img = raster.convert_img_to_array(img)[0]
     
     rvs = raster.get_image(root, img_type, rvs_name)
     rvs = raster.convert_img_to_array(rvs)[0]
     
     pyplot.hist(img, bins = 30, alpha=0.5, label='Roofs')
     pyplot.hist(rvs, bins = 30, alpha=0.5, label='Non-roofs')
     pyplot.legend(loc='upper right')
     
     pyplot.show()
     plt.savefig('/Users/akshatgoel/Desktop/test.png')

     
def main():
    '''
    --------------------------------------------------
    Input: 
    Output:
    --------------------------------------------------
    '''
    root = "Sentinel"
    img_type = "Gorakhpur"
    img_name = "RT_T44RQQ_20190521T045701_B02_mask.tif"
    rvs_name = "RT_T44RQQ_20190521T045701_B02_reverse_mask.tif"
    
    plot_histogram(root, img_type, img_name, rvs_name)

    
if __name__ == '__main__':
    main()
    