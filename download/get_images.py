# Import packages
import os 
import json
import boto3
import argparse
import numpy as np
import pandas as pd

# Import S3 wrappers
from helpers import common as helpers



def get_image_keys(pre):
    
    images = helpers.get_matching_s3_keys(prefix = pre)
    
    return(images)


def get_images(images):

    try: 
        for image in images:
            print('Downloading {}'.format(image))
            helpers.download_s3(image, './' + image)
    except Exception as e:
        print('Download error: {}'.format(e))
    
    
def get_input():
    
    # List of imagery here
    args = {1: 'Deoria Google Earth Image', 
            2: 'Deoria Landsat 30M',
            3: 'Deoria Metal Shapefile',
            4: 'Deoria NRSC 5M',
            5: 'Deroia Sentinel 10M',
            6: 'Ghaziabad GE Imagery'}
    
    # User inputs which imagery she wants
    print(args)
    arg = int(input("Enter what folder number you need from above:"))
    
    return(arg)

    
def main():
    
    arg = get_input()
    destination = args[arg]
    source, pre = get_paths(destination)
    
    make_folders(source)
    images = get_image_keys(pre)
    
    
    get_images(images)
    
    
if __name__ == '__main__':
    
    main()