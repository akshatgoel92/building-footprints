from predict import predict
from train import train
from unet import unet
    
    
def predict():
    
    try:
        predict.predict()
    except Exception as e:
        print(e)
    
    
def main():
    
    
    predict()
    
    
if __name__ == '__main__':
    main()