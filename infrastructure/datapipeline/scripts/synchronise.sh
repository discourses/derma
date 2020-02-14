#!/bin/bash
cd ~

# Delivering the results files to Amazon S3
namestring=`date +%Y%m%d.%H%M%S`
aws s3 sync ~/checkpoints/ s3://deep.learning.models.checkpoints/$namestring/
