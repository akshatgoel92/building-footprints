import numpy as np
import itertools
import random
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from helpers import common



def get_files(prefix, suffix):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    files = list(common.get_matching_s3_keys(prefix, suffix))
    random.seed(a=243, version=2)
    random.shuffle(files)
    
    return files


def get_train_dev_set(files, n, dev):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    flats = []
    
    if dev == 1:
        files_to_get = files[n:]
    else: 
        files_to_get = files[0:n]
    
    for f in files_to_get:
        flats.append(np.load(common.get_object_s3(f), allow_pickle = True)['arr_0'])
        
    return flats


def get_X_Y(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = [np.concatenate(a) for a in zip(*itertools.chain(df))]
    
    return(df)

    
def fit_log_reg(X, Y, C):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    log_reg = LogisticRegression(C=C, verbose = True, n_jobs = -1, solver = 'saga', warm_start = True)
    log_reg.fit(X, Y)
    
    return(log_reg)
    
    
def fit_random_forest(X, Y, n_estimators):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    rf_reg = RandomForestClassifier(n_estimators = n_estimators, max_depth = 2, 
                                    bootstrap = False, n_jobs = 1, 
                                    verbose = 1000, 
                                    warm_start = True)
    rf_reg.fit(X, Y)
    
    return(rf_reg)



def save_model(model, filename):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    joblib.dump(model, filename)


def get_predictions(model, X):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    Y_hat = model.predict(X)
    return(Y_hat)


def get_scores(model, X_test, Y_test):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    result = model.score(X_test, Y_test)
    return(result)