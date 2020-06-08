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

	TBC

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
