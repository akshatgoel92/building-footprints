# Overview

This is an experimental machine learning pipeline for semantic segmenation on Google Earth Engine data for Gorakhpur. The task we are trying to solve for is classifying all pixels which are part of a metal roof in a given satellite imagery tile from labeled data. The document given below summarizes each component of the pipeline including what it does and how to execute it. 

# download 

This module downloads existing imagery from S3 to the local machine. To run it just use the following command: 

```python 
python download/download.py 
```
