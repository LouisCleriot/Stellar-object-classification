#!/bin/bash

docker --version &> /dev/null
DOCKER_EXISTS=$?

if [ $DOCKER_EXISTS -ne 0 ]; then
    echo "Docker is installed."
    exit 0
else
    echo "Docker is not installed."
    exit 1
fi