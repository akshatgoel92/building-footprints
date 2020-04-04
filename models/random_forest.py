# Import packages
import numpy as np
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
from helpers import common


def fit(X, Y, n_estimators):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    rf_reg = RandomForestClassifier(n_estimators = n_estimators, max_depth = 2, bootstrap = False, n_jobs = 1, verbose = 1000)
    rf_reg.fit(X, Y)
    
    return(rf_reg)

def main():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    print("Running...")
    root = 'GE Gorakhpur'
    image_type = 'blocks'
    
    prefix = common.get_s3_paths(root, image_type)
    suffix = '.npz'
    n = 3
    run = 1
    n_estimators = 100
    
    
    files = get_files(prefix, suffix)
    print(files)
    
    train = get_train_set(files, n)
    train = execute_merge(train)
    
    X_train, Y_train = reshape_df(train)
    rf_reg = fit(X_train, Y_train, n_estimators)
    
    save_model(rf_reg, 'rf_reg_{}.sav'.format(str(run)))
    print("Done...!")
    
    
if __name__ == '__main__':
    main()