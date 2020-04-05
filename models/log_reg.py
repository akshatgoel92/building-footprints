from models import utils
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
    dev = 0
    
    files = utils.get_files(prefix, suffix)
    train = utils.get_train_dev_set(files, n, dev)
    print(train)
    
    X_train, Y_train = utils.get_X_Y(train)
    print(X_train.shape)
    print(Y_train.shape)
    
    hypers = [1]
    for c in hypers:
        log_reg = utils.fit_log_reg(X_train, Y_train, c)
        utils.save_model(log_reg, 'log_reg_{}.sav'.format(str(c)))
        
  
if __name__ == '__main__':
    main()