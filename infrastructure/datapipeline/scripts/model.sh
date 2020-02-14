#!/bin/bash
cd ~

# Runs the FeatureExtractionDL model.  It requires a string input argument; the string
# must be a URL oF A  YAML file of hyperparameters, e.g., 
# https://raw.githubusercontent.com/greyhypotheses/hub/develop/data/hyper/pattern.yml
sudo docker run -v ~/images:/app/images -v ~/checkpoints:/app/checkpoints greyhypotheses/derma:FeatureExtractionDL src/main.py $1
