import pandas as pd
import numpy as np

from summarize import utils
from helpers import raster
from helpers import vector


def main():

    # Convert to command line arguments
    # Need to remove hardcoding from this and other file
    bands = 3
    suffix = ".npz"
    root = "GE Gorakhpur"
    image_type = "blocks"
    
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--bands", type=int, default=bands)
    parser.add_argument("--suffix", type=str, default=suffix) 
    parser.add_argument("--image_type", type=str, default=image_type)
    
    args = parser.parse_args()
    root = args.root
    bands = args.bands
    suffix = args.suffix
    image_type = args.image_type
    
    # Get S3 paths
    # List files
    prefix = common.get_s3_paths(root, image_type)
    files = list(common.get_matching_s3_keys(prefix, suffix))

    for i, f in enumerate(files):

        # Load data
        df = utils.get_flat_file(f)

        # Get information
        area = get_poly_area(df)
        avgs = get_avgs(df)
        sd = get_sd(df)

        # Get summary dataframes
        df_area, df_summary = get_dfs(avgs, sd, area)

        # Call histograms
        # First make regular histogram
        df_regular = get_regular_data(df, bands)
        get_histogram(df_regular, f_no=i)

        # Now make overlaid histograms
        overlay = []

        for j, band in enumerate(range(bands)):
            df_overlay = get_overlay_data(df, band)
            get_histogram(df_overlay, f_no=j, overlay=1)

if __name__ == '__main__':
    main()