import joblib
import argparse

from models import utils


def main():

    
    filename = "models/results/log_reg_1.sav"
    prefix = "GE Gorakhpur"
    suffix = "blocks"
    dev = 1
    n = 3
    

    files = utils.get_files(prefix, suffix)
    dev = utils.get_train_dev_set(files, n, dev)

    if len(dev) > 1:
        X, Y = utils.get_X_Y(dev)
    elif len(dev) == 1:
        X,Y = utils.process_single(dev)

    X_dev, Y_dev = reshape_df(dev)
    model = joblib.load(filename)

    prediction = utils.get_predictions(model, X_dev)
    result = utils.get_scores(model, X_dev, Y_dev)


if __name__ == "__main__":
    main()
