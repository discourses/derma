#!/bin/bash
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo service docker start

# In order to use docker commands without 'sudo'
sudo usermod -a -G docker ec2-user

# Time
echo '"ZONE=\"GMT\""' > /etc/sysconfig/clock
ln -sf /usr/share/zoneinfo/GMT /etc/localtime