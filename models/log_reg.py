# Import packages
from sklearn.linear_model import LogisticRegression
from helpers import common
from models import utils

# 
def load_data(root, image_type):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    path = common.get_s3_paths(root, image_type)
    f = [get_object_s3(path, image) for image in images]
    


def merge_data():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    pass



def get_estimator(c):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    log_reg100 = LogisticRegression(C=100)
    log_reg100.fit(X_train, y_train)



def train():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    pass


def get_predictions(log_reg):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    pass


def get_scores(log_reg):
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    train_scores = []
    test_scores = []
    
    train_scores.append(log_reg.score(X_train, y_train))
    test_scores.append(log_reg.score(X_test, y_test))
    


def main():
    '''
    -------------------
    Input:
    Output:
    -------------------
    '''
    []



if __name__ == '__main__':
    main()

