import random
import math
import os
import re

from helpers import common


def shuffle_flat_files(prefix="GE Gorakhpur/tile", suffix=".tif"):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    random.seed(a=243, version=2)
    files = list(common.get_matching_s3_keys(prefix, suffix))

    random.shuffle(files)
    chunksize = math.ceil(len(files) / 4)
    chunks = range(0, len(files), chunksize)
    files = [files[x : x + chunksize] for x in chunks]

    return files


def get_train_val(files):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    train_frames = [item for sublist in files[0:3] for item in sublist]
    val_frames = files[-1]

    return (train_frames, val_frames)


def get_dest_files(train_target, val_target, train_frames, val_frames):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    train_dest_frames = [
        common.get_s3_paths(train_target, os.path.basename(img)) for img in train_frames
    ]
    val_dest_frames = [
        common.get_s3_paths(val_target, os.path.basename(img)) for img in val_frames
    ]

    return (train_dest_frames, val_dest_frames)


def add_frames(source_frames, dest_frames):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    counter = 0
    for source, destination in zip(source_frames, dest_frames):
        counter += 1
        print(counter)
        common.copy_object_s3(source, destination)
    return


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    root = "GE Gorakhpur"
    val_target = os.path.join(root, os.path.join("data", "val_frames"))
    train_target = os.path.join(root, os.path.join("data", "train_frames"))

    files = shuffle_flat_files()
    train_frames, val_frames = get_train_val(files)
    train_dest, val_dest = get_dest_files(
        train_target, val_target, train_frames, val_frames
    )
    print(train_dest)
    print(val_dest)

    add_frames(train_frames, train_dest)
    add_frames(val_frames, val_dest)


if __name__ == "__main__":
    main()
