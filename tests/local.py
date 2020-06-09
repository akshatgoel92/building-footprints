from chip import chip
from unet import utils
from unet import datagen

from train import train
from predict import predict
    
    
def run_test(func):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    def test(**args):
        
        try: 
            return(func(**args))
        
        except Exception as e:
            return(e)
            
    return(test)
        
    
@run_test
def test_chip(chip, args):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    print("Testing the chip module....")

@run_test
def test_predict():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(predict.main, args)
    
    
@run_test
def test_download():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(download.main(), args)
    
    
@run_test
def test_flatten():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(flatten.main(), args)
    
    
@run_test
def test_mask():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(mask.main(), args)
    
    
@run_test
def test_mosaic():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(mosaic.main, args)
    
    
@run_test
def test_summarize():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(mosaic.main, args)
    
@run_test
def test_split():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    run_test(split.main, args)
    
    
@run_test
def test_datagen(model_type="unet"):
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    settings = utils.get_settings(model_type)
    
    args = {'paths': utils.get_paths(**settings["path_args"]), 
            'load_dataset_args': settings["load_dataset_args"]}
    
    train, val = run_test(datagen.load_dataset, args)
    train_img = next(train)
    val_img = next(val)
    
    
if __name__ == "__main__":
    main()
