# Introduction
 

This is an experimental machine learning pipeline for semantic segmentation on Google Earth Engine data for Gorakhpur. The task we are trying to solve for is classifying all pixels which are part of a metal roof in a given satellite imagery tile from labeled data.

# Overview

These are the tasks in the order in which they have to be done to prepare the training data for both machine learning and deep learning: 

- Setup
- Download
- Chip
- Mosaic
- Mask 
- Split
- Flatten
- Summarize
- Train (Machine Learning)
  - Forest
  - Regression
- Train (Deep Learning) 
- Predict
- Tests

The following sections will describe what component of the repository does each of these tasks as well as how to run them.

### Hardware

Development: 60 GB RAM | 520 GB SSD | Ubuntu 18.04 | E2E Networks  
Production:  60 GB RAM | 520 GB SSD | Ubuntu 18.04 | NVidia Testla T4 GPU

### Setup 

The first step in the pipeline is setting up the environment. The pipeline is tested locally and then deployed on the cloud. We use the E2E cloud service. A computer with similar features can easily be found on Amazon Web Services or any other cloud provider. The first step to setting up is to start a new node on E2E. Once this is done log in to the node. Then clone this repository onto the server using git clone. Then run the following commands:
```
cd ~/roof-classify && chmod +x setup.sh && setup.sh
```
This will go to the roof-classify directory which was just created after you ran ``git clone ...`` and then will run the ``setup.sh`` shell script. If running this fails then open setup.sh and copy and paste each line into the server terminal. This will install all the required dependencies and create a functioning environment for you.

### Data Folder Structure 

     --- data
	       -- train_frames
		       -- train_0.tif
		       -- train_1.tif
		       -- train_2.tif
			       ...
		       -- train_40478.tif
	       -- train_masks
		       -- train_0.tif
		       -- train_1.tif
		       -- train_2.tif
			       ...
	       -- val_frames
         	       -- val_0.tif
		       -- val_1.tif
		       -- val_2.tif
			       ...
        -- val_masks
         	       -- val_0.tif
		       -- val_1.tif
		       -- val_2.tif
			       ...


### Download 

This module downloads existing imagery from S3 to the local machine. To run it just use the following command: 

```python 
python download/download.py 
```
This code will then give you a sequence of options to which you can reply to get the imagery that you need. This was used to download the Microsoft Bing Imagery but is not needed for SpaceNet2.

### Chip 

This module takes in a satellite imagery tile and chips it to a target file size. Whitespace is used as padding for the corners. To run it use the follow command: 

```python
python chip/chip.py
```

This code will then chip the tile to the desired file size and place the resulting chipped images in the target directory.

### Mosaic

This module takes in an input directory containing satellite imagery tiles of uniform size and uniform coordinate reference system and outputs a single mosaic tile in the target directory. To run it use the following command: 

```python 
python mosaic/mosaic.py
```

### Mask

This module takes in an input directory containing satellite imagery and the file path to a shapefile which contains polygon labels. It then creates grayscale masks by super-imposing the polygon labels on the satellite imagery and sends them to an output directory. To run this module use the following: 

```python 
python mask/mask.py 
```

### Flatten 

This module takes in an input directory containing satellite imagery and the file path to a target directory and returns the following collumns as a .csv file:
 - Latitude
 - Longitude
 - Row coordinate
 - Pixel coordinate
 - Band 
 - Pixel value 

To run this module use the following command: 
```python 
python flatten/flatten.py
```

### Summarize

This module takes in a flat .csv file with the columns listed above and returns histograms of band pixel values as .png files. To run this module use the following command: 
```python 
python summarize/summarize.py
```

### Split 

This module takes in a target directory with the folder structure given above and and splits the data into training and validation sets. To run this module use the following command: 
```python
python split/split.py
```

### Train (Deep Learning) 

This module trains a given deep neural network using keras. It takes in file paths to the following directories: 
 - Training images
 - Training masks
 - Validation images
 - Validation masks 

It uses settings.json to compile a keras model with these settings. Each setting is described below: 

```python 
    "path_args": {
        "train_frames": "train_frames",
        "train_masks": "train_masks", 
        "val_frames": "val_frames",
        "val_masks": "val_masks"
    }
```
These are the file paths to the data folders relative to the data storage folder in the folder structure given above. 

```python
    "checkpoint_args":{
        "checkpoint_path": "results_4.h5"
    }
```    
This is the path where you want to store the output weights after training. 

```python
    "extension_args": {
        "extension": ".tif"
    }
```
This is the file type of the data being used for training. 

```python
    "load_dataset_args":{
        "custom":1,
        "batch_size": 4,
        "target_size":[640, 640],
        "rescale": 255,
        "shear_range":0.2,
        "zoom_range":0.2,
        "horizontal_flip":true,
        "class_mode":"input",
        "mask_color":"grayscale",
        "channels":3, 
        "data_format": "channels_last"
    }
```
These are the preprocessing arguments for the data generating functions in the data generator. Further detail is given below: 
   - Custom: This gives which data generating function to use. The default KerasImageDataGenerator is used when this is set to 0.   
   - Batch size: This gives the batch size of the training data i.e. at each iteration of the optimizer how many images are used to update the weights. 
   - Target size: This gives the target size of the training data. 
   - Rescale: This gives the value by which to rescale the data. 
   - Shear range: This is an argument for Keras data generator so only gets used when custom = 0.
   - Zoom range: This is an argument for Keras data generator so only gets used when custom = 0. 
   - Horizontal flip: This is an argument for Keras data generator so only gets used when custom = 0. It flips each image horizontally in both directions and adds these flips to t                      he training dataset.
   - Class mode: This is an argument for Keras data generator so only gets used when custom = 0. It specifies that it should treat the images as input datasets and not raise an er                 ror when it finds no output images. 
   - Mask color: The image type of the mask. Generally set to grayscale.  
   - Channels: No. of channels or bands in the data. 
   - Data Format: This gives whether the imagery is in channels first (3, 640, 640) or channels last format (640, 640, 3).  


```python
    "model_args":{
        "input_shape":[640, 640, 3], 
        "num_classes":1, 
        "num_layers":4, 
        "filters": 64,
        "upconv_filters": 96, 
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
These are the arguments for the model. The parameters marked TBC are the ones that we are still unsure about. Further detail about this is given below: 
 - Input shape: This gives the shape of the input file.
 - Number of classes: This gives the number of possible output classes. This is set to 1 for segmentation but 2 for binary classification.
 - Number of layers: This gives the number of downsampling and upsampling layers for the U-Net. 
 - Upconvolution filters: [TBC]
 - Kernel Size: [TBC] 
 - Activation: This gives the activation function for each layer. 
 - Padding: This gives the padding in the convolutional layers. [TBC]
 - Initializer: This gives the weight initiliazation method for the start of training. 
 - Momentum: This gives the 'learning rate' or 'speed' of the optimizer. 
 - Pool size: This gives the size of the filter in the pooling layers. 
 - Pool strides: This gives the size of the stride in the pooling layers. 
 - Pool padding: This gives the padding in the pooling layers. 
   
```python 
    "training_args":{
        "epochs":50,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```
This block has the settings for training. Further detail on each argument is given below: 
 - Epoch: This gives the number of epochs. Each epoch is one pass through the training data. 
 - Pretrained: This gives whether we are using a pretrained model or not. 
 - Results Folder: This gives the folder where results are going to be stored. 
 - Steps per Epoch: It provides the number of steps per epoch for training. Right now it is manually calculated by No. of training images/Batch Size. 
 - Validation steps: It provides the number of steps per epoch for validation. Right now it is manually calculated by No. of validation images/Batch size.
 - Verbose: This tells keras to publish the metrics during training for monitoring.    

```python   
    "output_args":{
        "kernel_size":[1, 1],
        "strides":[1, 1],
        "activation":"sigmoid",
        "padding":"valid"
    }
}
```
This block has the settings for the output layer. 

To run this module adjust the settings.json file as required and then use the following command: 
```python
python train/train.py
```

### Predict

This module takes in a model .h5 file and an input directory containing images that we want to make predictions for and returns grayscale predicted masks. To run this module use the following command: 
```python 
python predict/predict.py
```

### Experiments/Problems

We have tried to use this pipeline for end-to-end training on SpaceNet2 data. We have not been successful in creating predicted building footprints. The pipeline outputs has returned blank masks in all our trainings. This section will go into detail about what we have tried to far. Our current work is focusing on finding what is causing our pipeline to predict blank masks and correcting this issue.

### Experiment 1 

We trained a UNET with the settings given below. The model arguments and output arguments are the default arguments in the keras-unet package which we have sourced the code from.

### Settings

```python
    "model_args":{
        "input_shape":[650, 650, 8],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":10,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```

### Results

The training failed. This is because KerasImageDataGenerator by default can only process imagery up to 4 bands of length and width that are multiples of 32. The SpaceNet Pan-Sharpened multi-band imagery is 8 bands and 650 X 650.  

### Changes 

We added the custom data generator given in unet/datagen.py which uses skimage to resize the images to 640 X 640. Skimage can take in 8 band imagery. 

### Experiment 2 

We trained a UNET with the settings given below. The model arguments and output arguments are the default arguments in the keras-unet package which we have sourced the code from and are the same as above. 

### Settings 

```python
    "model_args":{
        "input_shape":[640, 640, 8],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":10,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```

### Results 
The training worked but the loss metrics we were tracking at that time [IOU and F1 Score [Dice]] were both extremely tiny [0.0000e-4] and did not change even when we trained for 10 epochs.

### Changes 
We imported the IOU and F1 score from the unet-keras library to make sure that there was no bug. 

### Experiment 3
We trained a UNET with the settings given below. The model arguments and output arguments are the default arguments in the keras-unet package which we have sourced the code from a
nd are the same as above.

### Settings 

```python
    "model_args":{
        "input_shape":[640, 640, 8],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":10,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```
### Results 
We saw the same pattern as above. The training worked but the loss metrics we were tracking at that time [IOU and F1 Score [Dice]] were both extremely tiny [0.0000e-4] and did notchange even when we trained for 10 epochs.

### Changes
Experiment 3 made us think that the loss functions and metrics were fine but that there might be a problem with pre-processing. We checked the pre-processed images that were beingfed into the neural network from the custom data generator and realized that the code was being rescaled twice. The pixel values were rescaled from their original values to float values between 0 and 1 when ski-kit image resized them and then again when our rescale argument was applied. We removed the rescale argument from the custom image data generator at this point. 

### Experiment 4
We trained a UNET with the settings given below. 

### Settings 

```python
    "model_args":{
        "input_shape":[650, 650, 8],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":10,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```
### Results

The training worked fine. The IOU and F1 score as well as the loss function decreased with each epoch. They were all within the expected range. However when we tried to use the trained model weights to make predictions we got blank masks [all black with no footprints].

### Changes 
We thought that maybe we were not training for enough epochs. We changed the epochs setting to 50 and used early stopping. 

### Experiment 5 
We trained a UNET with the settings given below. 

### Settings 

```python
    "model_args":{
        "input_shape":[640, 640, 8],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":50,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```
### Results 
The training worked fine. The metrics behaved in the same way as above. Early stopping made training end after 43 epochs.  We were getting blank masks. When we tried to diagnose this problem by looking at the numerical values of the predictions we found that the predicted probabilities were very small. When we looked at the numerical pixel values going into the model post-resizing we found that they were also very small too. So the first possibility is that we are pre-processing our images incorrectly in the custom image data generator. The other possibility is that we are writing the masks wrong and that there is a bug in our prediction module. This is why the next experiment tries to isolate the problem. 

### Changes

To isolate the problem we decided to switch from 8-band Pan-Sharpened Multi-spectral images to 3-band RGB imagery. The reasoning behind this is given below. 

### Experiment 6

This experiment changes the input data to the SpaceNet2 RGB 3 band imagery. The reason this will help is because we have already tested the default image data generator using the Gorakhpur Microsoft Bing data. If this works fine then this indicates that the problem is restricted to the two possibilities mentioned above.

### Settings 

```python
    "model_args":{
        "input_shape":[640, 640, 3],
        "num_classes":1,
        "num_layers":4,
        "filters": 64,
        "upconv_filters": 96,
        "kernel_size": [3, 3],
        "activation": "relu",
        "strides": [1, 1],
        "padding": "same",
        "kernel_initializer": "he_normal",
        "bachnorm_momentum": 0.01,
        "pool_size":[2, 2],
        "pool_strides": [2, 2],
        "pool_padding": "valid"
    }
```
```python
    "training_args":{
        "epochs":50,
        "pretrained":false,
        "results_folder":"results",
        "steps_per_epoch":723,
        "validation_steps":241,
        "verbose":1
    }
```

### Results 

The default image data generator throws an error. The PIL image library keeps finding unidentified images in the dataset and is unable to load them. Further documentation on this error is given here. We tried to change the extension from .tif to .tiff as suggested in the post but this raised the same error. This seems to be a dataset issue rather than a code issue because this code has been tested on the Gorakhpur 4-band .tif imagery. My suggested next steps are to continue with diagnosing the problem in the custom image data generator and the footprint [prediction] creation code in further experiments rather than continuiing with this.

 
