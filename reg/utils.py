import numpy as np
import itertools
import random
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
    elif dev == 0:
        files_to_get = files[0:n]

    for f in files_to_get:
        flats.append(np.load(common.get_object_s3(f), allow_pickle=True)["arr_0"])

    return flats


def get_X_Y(df):
    """
    ------------------------
    Input: 
    Output:
    ------------------------
    """
    df = [np.concatenate(a) for a in zip(*itertools.chain(df))]

    return df


def get_X_Y_single(df):
    """
     -------------------
     Input:
     Output:
     -------------------
     """
    X = np.transpose(np.vstack(df[0][4:-2]))
    Y = np.transpose(np.array(df[0][-1].data))

    return (X, Y)


def get_nn_data(X, Y, length=256, width=256):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    size = length * width
    X_ = np.array([np.pad(x, (0, size - len(x) % size)) for x in X])
    X_ = [x.reshape(len(x) // size, length, width) for x in X_]


def fit_log_reg(X, Y, C):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    log_reg = LogisticRegression(C=C, verbose=True, n_jobs=-1, solver="saga", warm_start=True)
    log_reg.fit(X, Y)

    return log_reg


def fit_random_forest(X, Y, n_estimators):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    rf_reg = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, bootstrap=False, n_jobs=1, verbose=1000, warm_start=True,)
    rf_reg.fit(X, Y)

    return rf_reg


def save_model(model, filename):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    joblib.dump(model, filename)


def get_predictions(model, X):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    Y_hat = model.predict(X)
    return Y_hat


def get_confusion_matrix(model, Y, Y_pred):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    return confusion_matrix(Y, Y_pred)


def get_scores(model, X_test, Y_test):
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    result = model.score(X_test, Y_test)
    return result


def get_other_scores():
    """
    -------------------
    Input:
    Output:
    -------------------
    """
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    return (tn, fp, fn, tp, sens, spec)
