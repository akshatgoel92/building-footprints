from predict import predict
from train import train

from unet import datagen
from unet import utils


def test_predict():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    try:
        predict.main(test=1)

    except Exception as e:
        print("Got an error!")
        print(e)


def test_datagen(model_type="unet"):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    settings = utils.get_settings(model_type)
    load_dataset_args = settings["load_dataset_args"]

    path_args = settings["path_args"]
    paths = utils.get_paths(**path_args)
    train, val = datagen.load_dataset(paths, load_dataset_args)

    try:
        train_img = next(train)
    except Exception as e:
        print("Got an error!")
        print(e)

    try:
        val_img = next(val)
    except Exception as e:
        print("Got an error!")
        print(e)


def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    test_predict()
    test_datagen()


if __name__ == "__main__":
    main()
