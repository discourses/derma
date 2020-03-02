#!/bin/bash
cd ~

# Runs the FeatureExtractionDL model.  It requires one string argument; the string
# must be a URL oF A  YAML file of hyperparameters, e.g.,
# https://raw.githubusercontent.com/greyhypotheses/dictionaries/develop/derma/hyperparameters/pattern.yml
sudo docker run -v ~/images:/app/images -v ~/checkpoints:/app/checkpoints greyhypotheses/derma:FeatureExtractionDL src/main.py $1
