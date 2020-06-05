import shutil
import random
import math
import os
import re

from helpers import common


def shuffle_files(files):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    random.seed(a=243, version=2)
    random.shuffle(files)

    return files


def get_train_val(files, train_split=0.75):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    last_img_train = math.ceil(train_split * len(files))
    train_frames = files[0:last_img_train]
    val_frames = files[last_img_train:]

    return (train_frames, val_frames)


def get_dest_files(train_target, val_target, train_frames, val_frames):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    train_dest_frames = [os.path.join(train_target, os.path.basename(img)) for img in train_frames]
    val_dest_frames = [os.path.join(val_target, os.path.basename(img)) for img in val_frames]

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
        shutil.copyfile(source, destination)
    return


def main():
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    root = "data"
    image_type = "frames"

    train_target = os.path.join(root, "train_frames")
    val_target = os.path.join(root, "val_frames")

    prefix = common.get_local_image_path(root, image_type)
    files = [os.path.join(prefix, f) for f in common.list_local_images(root, image_type)]

    files = shuffle_files(files)
    train_frames, val_frames = get_train_val(files)
    train_dest, val_dest = get_dest_files(train_target, val_target, train_frames, val_frames)

    add_frames(train_frames, train_dest)
    add_frames(val_frames, val_dest)


if __name__ == "__main__":
    main()
