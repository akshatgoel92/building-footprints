# Import packages
import os 
import json
import boto3
import argparse
import numpy as np
import pandas as pd
    
    
def get_object_s3(key):
    '''
    Get an object from S3.
    '''
    
    with open('./secrets.json') as secrets:
        s3_access = json.load(secrets)['s3']
    
    bucket_name = s3_access['default_bucket']
    access_key_id = s3_access['access_key_id']
    secret_access_key = s3_access['secret_access_key']
    
    s3 = boto3.client('s3',access_key_id, secret_access_key)
    response = s3.get_object(bucket_name, key)
    file = response["Body"]
    
    return(file)
    
    
def upload_s3(file_from, file_to):
    '''
    Upload an object to S3.
    '''
    
    with open('./secrets.json') as secrets:
        s3_access = json.load(secrets)['s3']
    
    bucket_name = s3_access['default_bucket']
    access_key_id = s3_access['access_key_id']
    secret_access_key = s3_access['secret_access_key']
    
    s3 = boto3.client('s3', access_key_id, secret_access_key)
    s3.upload_file(file_from, bucket_name, file_to)
    
    
def download_file_s3(file_from = '/Image Folder/Deoria Google Earth Image/AOI_GE_Image_1_1.tif',
                     file_to = './AOI_GE_Image_1_1.tif'):
    '''
    Download a particular file from S3.
    '''
    
    with open('./secrets.json') as secrets:
        s3_access = json.load(secrets)['s3']
    
    bucket_name = s3_access['default_bucket']
    access_key_id = s3_access['access_key_id']
    secret_access_key = s3_access['secret_access_key']
    
    s3 = boto3.resource('s3', 
                        aws_access_key_id = access_key_id, 
                        aws_secret_access_key = secret_access_key)
    try:
        s3.Bucket(bucket_name).download_file(file_from, file_to)
    except Exception as e: 
        print(e)
    
    
def get_matching_s3_objects(prefix="", suffix=""):
    """
    Generate objects in an S3 bucket.
    :param prefix: Only fetch objects whose key starts with this prefix (optional).
    :param suffix: Only fetch objects whose keys end with this suffix (optional).
    Taken from: https://alexwlchan.net/2019/07/listing-s3-keys/
    Copyright © 2012–19 Alex Chan. Prose is CC-BY licensed, code is MIT.
    """
    with open('./secrets.json') as secrets:
        s3_access = json.load(secrets)['s3']
        
    bucket_name = s3_access['default_bucket']
    access_key_id = s3_access['access_key_id']
    secret_access_key = s3_access['secret_access_key']
    
    s3 = boto3.client("s3", access_key_id, secret_access_key)
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {'Bucket': bucket_name}
    	
    # We can pass the prefix directly to the S3 API.  
    # If the user has passed a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str): prefixes = (prefix, )
    else: prefixes = prefix
    
    for key_prefix in prefixes: 
        kwargs["Prefix"] = key_prefix
        
        for page in paginator.paginate(**kwargs):
            try: contents = page["Contents"]
            except Exception as e: print(e) 
            
            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj
    
    
def get_matching_s3_keys(prefix="", suffix=""):
    """
    Generate the keys in an S3 bucket.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    Taken from: https://alexwlchan.net/2019/07/listing-s3-keys/
    Copyright © 2012–19 Alex Chan. Prose is CC-BY licensed, code is MIT.
    """
    for obj in get_matching_s3_objects(prefix, suffix): yield obj["Key"]
    
    
if __name__ == '__main__':
    
    main()
    