#!/bin/bash
cd ~
aws s3 sync ~/checkpoints/ s3://deep.learning.models.checkpoints/$namestring/
