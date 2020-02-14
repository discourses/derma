#!/bin/bash
cd ~

# Import the model 'docker run' script
aws s3 cp s3://engineering.infrastructure.definitions/projects/derma/infrastructure/datapipeline/scripts/model.sh .
