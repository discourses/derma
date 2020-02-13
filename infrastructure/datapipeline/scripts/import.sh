#!/bin/bash
cd ~
docker run -d -v ~/images:/app/images greyhypotheses/derma:importing
