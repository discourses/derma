#!/bin/bash
cd ~

# Stop all containers
sudo docker stop $(sudo docker ps -a -q)

# Remove all containers and associated volumes
sudo docker rm -v $(sudo docker ps -a -q)

# Remove all images
sudo docker rmi $(sudo docker images -a -q)

# Delete all project files & directories
sudo rm -rf checkpoints/
sudo rm -rf images/
