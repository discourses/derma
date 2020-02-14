#!/bin/bash
cd ~

# Delivering the results files to Amazon S3
aws s3 sync ~/checkpoints/ s3://deep.learning.models.checkpoints/$namestring/
