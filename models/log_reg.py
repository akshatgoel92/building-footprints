# Import packages
import numpy as np
import random
import joblib
from sklearn.linear_model import LogisticRegression
from helpers import common


def get_flat_files(prefix, suffix, n):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    files = list(common.get_matching_s3_keys(prefix, suffix))
    random.seed(a=243, version=2)
    random.shuffle(files)
    flats = []
    dev = files[n:]
    
    for f in files[0:n]:
        flats.append(np.load(common.get_object_s3(f), allow_pickle = True)['arr_0'])
    
    
    return flats, dev


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
    df = np.transpose(np.array(df))
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

def process_single_dev(dev):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    df_dev = []
    # Assume that the code has the mask at the end
    for a in dev[0][4:-2]:
        df_dev.append(a)

    df_dev.append(dev[0][-1].data)
    df_dev = np.transpose(np.array(df_dev))
    
    return(df_dev)


def stack_vertical(df):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    X = df[:,:3]
    Y = df[:,-1]
    
    return(X, Y)


def train(X, Y, C):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    log_reg = LogisticRegression(C=C)
    log_reg.fit(X, Y)
    
    return(log_reg)


def save_model(model, filename):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    joblib.dump(model, filename)

def get_predictions(log_reg, X):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    Y_hat = log_reg.predict(X)
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
    
    train, dev = get_flat_files(prefix, suffix, n)
    df_train = execute_merge(train)
    
    if len(dev) > 1:
        df_dev = execute_merge(dev)
    else:
        df_dev = process_single_dev(dev)
    
    X_train, Y_train = stack_vertical(df_train)
    X_dev, Y_dev = stack_vertical(df_dev)
    
    hypers = [1, 10]
    
    for c in hypers:
        log_reg = train(X_train, Y_train, c)
        save_model(log_reg, 'log_reg_{}.sav'.format(str(c)))
    
    print("Done...!")
    
    '''
    results = []
    for c in hypers:
        results.append(get_scores('log_reg_{}.sav'.format(c), X_dev, Y_dev))
    '''
        
    
if __name__ == '__main__':
    main()
    
    
    
    