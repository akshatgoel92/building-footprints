from results import utils
from helpers import common
from sklearn.linear_model import LogisticRegression


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
    
    files = utils.get_files(prefix, suffix)
    train = utils.get_train_set(files, n)
    train = utils.execute_merge(train)
    X_train, Y_train = utils.reshape_df(train)
    print(X_train.shape)
    print(Y_train.shape)
    
    hypers = [1]
    for c in hypers:
        log_reg = utils.fit_log_reg(X_train, Y_train, c)
        utils.save_model(log_reg, 'log_reg_{}.sav'.format(str(c)))
        
  
if __name__ == '__main__':
    main()