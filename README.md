# Overview

This is an experimental machine learning pipeline for semantic segmenation on Google Earth Engine data for Gorakhpur. The task we are trying to solve for is classifying all pixels which are part of a metal roof in a given satellite imagery tile from labeled data.

# Overview

These are the tasks in the order in which they have to be done to prepare the training data for both machine learning and deep learning: 

- Setup
- Download
- Chip
- Mosaic
- Fold
- Flatten
- Summarize
- Train (Machine Learning)
- Train (Deep Learning) 

The following sections will describe what component of the repository does each of these tasks as well as how to run them.

## Setup 

The first step in the pipeline is setting up the environment. The pipeline is tested locally and then deployed on the cloud. We use the E2E cloud service. We are using a 60 GB RAM 520 GB HD Ubuntu 18.04 Memory-optimized Machine for this pipeline. A computer with similar features can easily be found on Amazon Web Services or any other cloud provider. The first step to setting up is to start a new node on E2E. Once this is done log in to the node. Then clone this repositoryonto the server using git clone. Then run the following commands:
```
cd ~/roof-classify && chmod +x setup.sh && setup.sh
```
This will go to the roof-classify directory which was just created after you ran ``git clone ...`` and then will run the ``setup.sh`` shell script. If running this fails then open setup.sh and copy and paste each line into the server terminal. This will install all the required dependencies and create a functioning environment for you.


### download 

This module downloads existing imagery from S3 to the local machine. To run it just use the following command: 

```python 
python download/download.py 
```
