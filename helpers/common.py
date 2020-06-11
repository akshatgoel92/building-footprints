# Import packages
import io
import os
import json
import boto3
import argparse
import numpy as np


def get_local_folder_path(root, image_type):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    return os.path.join(root, image_type)


def get_local_image_path(root, image_type, image_name=""):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    return os.path.join(root, image_type, image_name)


def list_local_images(root, image_type, suffix = ".tif"):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    path = get_local_folder_path(root, image_type)

    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(suffix)]

    return images


def get_s3_paths(root, image_type):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    source = os.path.join(root, image_type)

    return source


def make_folder(name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    try:
        os.makedirs(name)
    except Exception as e:
        print(e)
        pass


def make_folders(root, image_type):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    make_folder(root)
    make_folder(image_type)


def get_remaining(
    output_format, extension, storage, prefix, prefix_storage,
):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    files = [img for img in os.listdir(prefix) if img.endswith(extension)]

    try:

        existing = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(prefix_storage) if f.endswith(output_format)]
        remaining = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in existing]

    except Exception as e:
        print(e)
        existing = []
        remaining = [f for f in files if os.path.splitext(os.path.basename(f))[0]]

    return remaining


def get_credentials():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    with open("./helpers/secrets.json") as secrets:
        s3_access = json.load(secrets)["s3"]

    return (
        s3_access["default_bucket"],
        s3_access["access_key_id"],
        s3_access["secret_access_key"],
    )


def get_s3_client():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    _, access_key_id, secret_access_key = get_credentials()
    s3 = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    return s3


def get_s3_resource():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    _, access_key_id, secret_access_key = get_credentials()
    s3 = boto3.resource("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    return s3


def get_bucket_name():
    """
    ------------------------
    Input: None
    Output S3 bucket name
    ------------------------
    """
    bucket_name, _, _ = get_credentials()
    return bucket_name


def copy_object_s3(source, destination):
    """
    ------------------------
    Input: None
    Output S3 bucket name
    ------------------------
    """
    s3_resource = get_s3_resource()
    bucket_name = get_bucket_name()

    source = os.path.join(bucket_name, source)
    destination = destination

    s3_resource.Object(bucket_name, destination).copy_from(CopySource=source)


def put_object_s3(f, key):
    """
    ------------------------
    Input: None
    Output S3 bucket name
    ------------------------
    """
    client = get_s3_client()
    bucket_name = get_bucket_name()
    response = client.put_object(Bucket=bucket_name, Body=f, Key=key)


def get_matching_s3_objects(prefix="", suffix=""):
    """
    ------------------------
    Generate objects in an S3 bucket.
    :param prefix: Only fetch objects whose key starts with this prefix (optional).
    :param suffix: Only fetch objects whose keys end with this suffix (optional).
    Taken from: https://alexwlchan.net/2019/07/listing-s3-keys/
    Copyright © 2012–19 Alex Chan. Prose is CC-BY licensed, code is MIT.
    ------------------------
    """
    s3 = get_s3_client()
    kwargs = {"Bucket": get_bucket_name()}
    paginator = s3.get_paginator("list_objects_v2")

    if isinstance(prefix, str):
        prefixes = (prefix,)

    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):

            try:
                contents = page["Contents"]
            except Exception as e:
                print(e)

            for obj in contents:
                key = obj["Key"]

                if key.endswith(suffix):
                    yield obj


def get_matching_s3_keys(prefix="", suffix=""):
    """
    ------------------------
    Generate the keys in an S3 bucket.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    Taken from: https://alexwlchan.net/2019/07/listing-s3-keys/
    Copyright © 2012–19 Alex Chan. Prose is CC-BY licensed, code is MIT.
    ------------------------
    """
    for obj in get_matching_s3_objects(prefix, suffix):
        yield obj["Key"]


def get_object_s3(key):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    s3 = get_s3_client()
    bucket_name = get_bucket_name()
    f = io.BytesIO(s3.get_object(Bucket=bucket_name, Key=key)["Body"].read())

    return f


def download_s3(file_from, file_to):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    s3 = get_s3_resource()
    bucket_name = get_bucket_name()

    try:
        s3.Bucket(bucket_name).download_file(file_from, file_to)

    except Exception as e:
        print(e)


def upload_s3(file_from, file_to):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    s3 = get_s3_client()
    bucket_name = get_bucket_name()
    s3.upload_file(file_from, bucket_name, file_to)


def get_raster_from_s3(path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """

    bucket_name, access_key, secret_access_key = common.get_credentials()

    url = "s3://{}/{}".format(bucket_name, path)

    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)

    with rasterio.Env(AWSSession(session)):
        f = rasterio.open(url)

    return f


def upload_mask(root, image_type, image_name):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    _, access_key, secret_access_key = common.get_credentials()

    s3_folder = common.get_s3_paths(root, image_type)
    file_to = os.path.join(s3_folder, image_name)
    common.upload_s3(file_from, file_to)


def upload_chips(in_path, out_path):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    upload_files = [os.path.join(out_path, img) for img in common.list_local_images(out_path, "")]

    for img in upload_files:
        file_to = common.get_s3_paths(in_path, out_path)
        common.upload_s3(img, os.path.join(in_path, img))


def upload_flat_file(flat, root, image_type, image_name):
    """
        ------------------------
        Input: 
        Output:
        ------------------------
        """
    file_from = common.get_local_image_path(root, image_type, image_name)
    np.savez_compressed(file_from, flat)

    _, access_key, secret_access_key = common.get_credentials()

    s3_folder = common.get_s3_paths(root, image_type)
    file_to = os.path.join(s3_folder, image_name)
    common.upload_s3(file_from, file_to)
