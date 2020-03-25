from matplotlib import pyplot as plt
from rasterio.plot import show_hist
from matplotlib import pyplot
from helpers import raster
from helpers import common 

import os
import json
import argparse
import rasterio


def get_hist_name(root, image_type, image_name):
    '''
    --------------------------------------------------
    Input: 
    Output:
    --------------------------------------------------
    ''' 
    # Now we split the text
    name = os.path.splitext(image_name)[0] + '_hist.png'
    partial = os.path.join(image_type, name)
    path = os.path.join(root, partial)
    
    return(path)


def plot_histogram_overlaid(root, image_type, image_name):
    '''
    --------------------------------------------------
    Input: 
    Output:
    --------------------------------------------------
    '''        
    # Store all histogram arguments here
    hist_args = {
                'histtype': 'stepfilled',
                'bins': 500,
                'alpha': 0.3, 
                }
    
    # Store the colors here
    # The max. bands across the imagery we have is six 
    # That's why we have chosen six values here
    colors = ['red', 'green', 'blue', 
              'yellow', 'purple', 'orange']
    
    # Store annotations here
    title = "Pixel value distribution"
    ylabel = "Normalized probability"
    xlabel = "Value"
    
    # Set up the figure object
    ax = plt.gca()
    fig = ax.get_figure()
    
    # Compute the values and weights
    img = raster.get_image(root, image_type, image_name)
    
    # Now iterate through bands and draw each layer
    for band in range(1, img.count + 1):
        
        # Now prepare image
        band_data = np.unique(img.read(band), return_counts = True)
        indices = np.where(band_data[0] > 0)
    
        # Store the values and weights
        vals = band_data[0][indices]
        
        # Note that we are normalizing to probabilities
        weights = band_data[1][indices]
        weights = weights/np.sum(weights)
        
        ax.hist(vals, 
                weights = weights, 
                label = str(band),
                color = colors[band - 1],
                range = (0, 0.4),
                **hist_args)
    
    # Add annotations
    ax.grid(True)    
    ax.legend(loc="upper right")
    ax.set_title(title, fontweight='bold')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
     
    name = get_hist_name(root, image_type, image_name)
    plt.savefig(name)
    
             
def main():
    
    parser = argparse.ArgumentParser()
    parse.add_argument('--img_type', type = str, required = False, default = 'Gorakhpur')
    parse.add_argument('--suffix', type = str, required = False, default = 'stacked')
    parse.add_argument('--root', type = str, required = False, default = 'Sentinel')
    
    args = parser.parse_args()
    img_type = args.img_type
    suffix = args.suffix
    root = args.root
    
    args = {
        
        'root': root,
        'image_type': img_type,
        'image_name': "RT_T44RQQ_20190521T045701_B02_{}.tif".format(suffix)
    
    }
    
    plot_histogram_overlaid(**args)

            
if __name__ == '__main__':
    main()