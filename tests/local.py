from predict import predict
    
    
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
    
    
def main():
    """
    ---------------------------------------------
    Input: None
    Output: None
    Run the test harness for evaluating a model
    ---------------------------------------------
    """
    test_predict()
    
    
if __name__ == '__main__':
    main()