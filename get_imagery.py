# Import packages
import os 
import json
import boto3
import argparse
import numpy as np
import pandas as pd

# Import S3 wrappers
import helpers



def make_image_folder():
    
    helpers.make_folder('./Image Folder')


def get_imagery(destination):
    
    source = "Image Folder/" + destination
    helpers.make_folder('./' + source)
    
    files = helpers.get_matching_s3_keys(prefix = source)
    for image in files:
        helpers.download_s3(image, destination + image)


def get_google_earth():
    
    helpers.make_folder('./Deoria Google Earth Image')


def get_landsat_30m():
    
    helpers.make_folder('./Deoria Landsat 30M')


def get_metal_shapefile():
    
    helpers.make_folder('./Deoria Metal Shapefile')


def get_nrsc_5m():
    
    helpers.make_folder('./Deoria NRSC 5M')


def get_sentinel_10m():
    
    helpers.make_folder('./Deoroia Sentinel 10M')


def get_ghaziabad_ge():
    
    helpers.make_folder('./Ghaziabad GE Imagery')


def main():
    
    pass


if __name__ '__main__':
    
    main()