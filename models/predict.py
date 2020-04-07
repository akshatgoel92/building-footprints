import joblib
import argparse

from helpers import common
from models import utils


def main():

    
    filename = "models/results/log_reg_1.sav"
    root = "GE Gorakhpur"
    image_type = "blocks"
    
    filename = "models/results/log_reg_1.sav"
    prefix = common.get_s3_paths(root, image_type)
    suffix = ".npz"
    dev = 1
    n = 3
    

    files = utils.get_files(prefix, suffix)
    print(files)
    dev = utils.get_train_dev_set(files, n, dev)
    print(dev)

    if len(dev) > 1:
        X, Y = utils.get_X_Y(dev)
    elif len(dev) == 1:
        X,Y = utils.process_single(dev)
        
    model = joblib.load(filename)
    
    prediction = utils.get_predictions(model, X)
    result = utils.get_scores(model, X, Y)


if __name__ == "__main__":
    main()
