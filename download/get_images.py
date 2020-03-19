# Import packages
import os 
import json
import boto3
import argparse
import numpy as np
import pandas as pd

# Import S3 wrappers
from helpers import common


def get_image_keys(pre):
    
    images = common.get_matching_s3_keys(prefix = pre)
    
    return(images)


def get_images(images):

    try: 
        
        for image in images:
            
            print('Downloading {}'.format(image))
            common.download_s3(image, image)
    
    except Exception as e:
        
        print('Download error: {}'.format(e))
    
def get_root_folder():
    
    # List of folders on S3 here
    folders = {1: 'Image Folder', 
               2: 'Sentinel'}
    
    # User inputs which folder she wants
    print(folders)
    arg = int(input("Enter what S3 folder number you need from above: "))
    
    folder = folders[arg]
    
    return(folder)
    
def get_image_type():
    
    # List of imagery here
    args = {1: 'Deoria Google Earth Image', 
            2: 'Deoria Landsat 30M',
            3: 'Deoria Metal Shapefile',
            4: 'Deoria NRSC 5M',
            5: 'Deroia Sentinel 10M',
            6: 'Ghaziabad GE Imagery',
            7: 'Gorakhpur'}
    
    # User inputs which imagery she wants
    print(args)
    arg = int(input("Enter what image type number you need from above:"))
    
    destination = args[arg]
    
    return(destination)

    
def main():
    
    root = get_root_folder()
    destination = get_image_type()
    source, pre = common.get_s3_paths(destination)
    
    common.make_folders(source, root)
    images = get_image_keys(pre)
    
    # Now we get the actual images
    get_images(images)
    
    
if __name__ == '__main__':
    
    main()