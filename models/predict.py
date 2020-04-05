import joblib
import argparse

from models import utils


def main():
    
    n = 3
    filename = 'models/results/rf_reg_1.sav'
    
    files = utils.get_files(prefix, suffix)
    dev = utils.get_dev_set(files, n)
    
    if len(dev) > 1:
        dev = utils.execute_merge(dev)
    elif len(dev) == 1:
        dev = utils.process_single(dev)
    
    X_dev, Y_dev = reshape_df(dev)
    model = joblib.load(filename)
    
    prediction = utils.get_predictions(model, X_dev)
    result = utils.get_scores(model, X_dev, Y_dev)
    

if __name__ == '__main__':
    main()