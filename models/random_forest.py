# Import packages
import numpy as np
import random
import joblib

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


def get_train_set(files, n):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    flats = []
    for f in files[0:n]:
        flats.append(np.load(common.get_object_s3(f), allow_pickle = True)['arr_0'])
        
    return flats


def get_dev_set(files, n):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    dev = []
    for f in files[n:]:
        dev.append(np.load(common.get_object_s3(f), allow_pickle = True)['arr_0'])
    
    return(dev)


def merge_flat_file(df1, df2):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = []
    # Assume that the code has the mask at the end
    for a, b in zip(df1[4:-2], df2[4:-2]):
        df.append(np.concatenate((a, b)))
    
    df.append(np.concatenate((df1[-1].data, df2[-1].data)))
    
    return(df)
    
    
def execute_merge(files):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    df = files[0]
        
    for i in range(len(files)-1):
        print(i)
        df1 = files[i+1]
        df = merge_flat_file(df, df1)
    
    return(df)

def process_single(df):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    df_res = []
    # Assume that the code has the mask at the end
    for a in df[0][4:-2]:
        df_res.append(a)

    df_res.append(df[0][-1].data)
    
    return(df_res)


def reshape_df(df):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    df = np.transpose(np.array(df))
    X = df[:,:3]
    Y = df[:,-1]
    
    return(X, Y)


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


def get_scores(filename, X_test, Y_test):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X_test, Y_test)
    return(result)

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
    n_estimators = 50
    
    files = get_files(prefix, suffix)
    print(files)
    
    train = get_train_set(files, n)
    train = execute_merge(train)
    
    X_train, Y_train = reshape_df(train)
    rf_reg = fit(X_train, Y_train, n_estimators)
    
    save_model(rf_reg, 'rf_reg_{}.sav'.format(str(c)))
    print("Done...!")
    
    
if __name__ == '__main__':
    main()