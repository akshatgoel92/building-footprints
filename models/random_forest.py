# Import packages
import numpy as np
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
from helpers import common


def main():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    root = 'GE Gorakhpur'
    image_type = 'blocks'
    
    prefix = common.get_s3_paths(root, image_type)
    suffix = '.npz'
   
    n = 3
    dev = 0
    run = 1
    n_estimators = 100
    files = utils.get_files(prefix, suffix)
    
    for f in files[0:n]:
        train = utils.get_train_dev_set(files = [f], n = 1, dev = dev)
        train = utils.get_X_Y(train)
        rf_reg = utils.fit_random_forest(np.transpose(np.array(train[4:-2])), np.transpose(np.array(train[-1].data)), n_estimators)
        
    utils.save_model(log_reg, 'rf_reg_{}.sav'.format(str(run)))


if __name__ == '__main__':
    main()