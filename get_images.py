# Import packages
import os 
import json
import boto3
import argparse
import numpy as np
import pandas as pd

# Import S3 wrappers
from helpers import common as helpers


def get_paths(destination, root = "Image Folder"):
    
    source = "./" + root + '/' + destination
    pre = root + '/' + destination
    
    return(source, pre)


def make_folders(source, root = "Image Folder",):
    
    helpers.make_folder('./' + root)
    helpers.make_folder(source) 


def get_image_keys(pre):
    
    images = helpers.get_matching_s3_keys(prefix = pre)
    
    return(images)


def get_images(images):
    '''
    Then downloads the images
    '''
    try: 
        for image in images:
            print('Downloading {}'.format(image))
            helpers.download_s3(image, './' + image)
    except Exception as e:
        print('Download error: {}'.format(e))


def main():
    
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
    
    # Run the data download pipeline
    destination = args[arg]
    source, pre = get_paths(destination)
    
    make_folders(source)
    images = get_image_keys(pre)
    
    # Finally we get the images here
    get_images(images)
    
    
if __name__ == '__main__':
    
    main()