#!/bin/bash
docker run -v ~/images:/app/images -v ~/checkpoints:/app/checkpoints greyhypotheses/derma:FeatureExtractionDL src/main.py $1