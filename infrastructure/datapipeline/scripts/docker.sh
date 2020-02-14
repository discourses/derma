#!/bin/bash

# Update instance libraries
sudo yum update -y

# Install Docker
sudo yum install -y docker

# Start docker
sudo service docker start
