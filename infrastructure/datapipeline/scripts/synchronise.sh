#!/bin/bash
cd ~

# Delivering results to Amazon S3
aws s3 sync ~/checkpoints/ s3://deep.learning.models.checkpoints/$namestring/
