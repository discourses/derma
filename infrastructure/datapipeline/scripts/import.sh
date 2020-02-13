#!/bin/bash
cd ~
sudo docker run -d -v ~/images:/app/images greyhypotheses/derma:importing
