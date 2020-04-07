import os
import joblib
import argparse

from helpers import common
from models import utils


def main():
    
    name = "log_reg_1.sav"
    root = "GE Gorakhpur"
    image_type = "blocks"
    
    filename = os.path.join(os.path.join("models", "results"), name)
    prefix = common.get_s3_paths(root, image_type)
    suffix = ".npz"
    dev = 1
    n = 3
    

    files = utils.get_files(prefix, suffix)
    print(files)
    
    dev = utils.get_train_dev_set(files, n, dev)
    
    if len(dev) > 1:
        X, Y = utils.get_X_Y(dev)
    elif len(dev) == 1:
        X,Y = utils.get_X_Y_single(dev)
        
    result = []    
    model = joblib.load(filename)
    result.append(utils.get_scores(model, X, Y))
    prediction = utils.get_predictions(model, X)
    confusion = get_confusion_matrix(model, Y, prediction)
    tn, fp, fn, tp, sens, spec = get_other_scores(confusion)
    
    result.append(tn)
    result.append(fp)
    result.append(fn)
    result.append(tp)
    result.append(sens)
    result.append(spec)
    
    print(confusion)
    print(result)

if __name__ == "__main__":
    main()