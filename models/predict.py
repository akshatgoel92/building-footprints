import numpy as np
import random
import joblib

from sklearn.linear_model import LogisticRegression
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


def get_predictions(model, X):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    Y_hat = model.predict(X)
    return(Y_hat)


def get_scores(model, X_dev, Y_dev):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    
    result = model.score(X_test, Y_test)
    return(result)

def main():
    filename = ''
    loaded_model = joblib.load(filename)


if __name__ == '__main__':
    main()