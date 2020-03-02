#!/bin/bash
cd ~
mkdir images
mkdir checkpoints

# Import greyhypotheses/derma:importing from Docker Hub.
# https://hub.docker.com/r/greyhypotheses/derma/tags
sudo docker pull greyhypotheses/derma:importing

# Running docker package greyhypotheses/derma:importing
sudo docker run -v ~/images:/app/images greyhypotheses/derma:importing

# Import greyhypotheses/derma:FeatureExtractionDL from Docker Hub.
# https://hub.docker.com/r/greyhypotheses/derma/tags
sudo docker pull greyhypotheses/derma:FeatureExtractionDL

# Runs the FeatureExtractionDL model.  It requires one string argument; the string
# must be a URL oF A  YAML file of hyperparameters, e.g.,
# https://raw.githubusercontent.com/greyhypotheses/dictionaries/develop/derma/hyperparameters/pattern.yml
sudo docker run -v ~/images:/app/images -v ~/checkpoints:/app/checkpoints greyhypotheses/derma:FeatureExtractionDL src/main.py $1

# Delivering the results files to Amazon S3
namestring=`date +%Y%m%d.%H%M%S%6N`
aws s3 sync ~/checkpoints/ s3://models.checkpoints/$namestring/
