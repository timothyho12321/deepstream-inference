#!/bin/bash

# Script to build and run Fish Monitoring App on Jetson Orin
# Requires sudo privileges for docker

echo "[INFO] Checking for NVIDIA Runtime..."
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "[ERROR] NVIDIA Container Runtime not found!"
    echo "       Please install nvidia-container-toolkit and configure docker."
    exit 1
fi

echo "[INFO] Allowing X11 access (for display if attached)..."
xhost + > /dev/null 2>&1 || true

echo "[INFO] Building Docker Image (this may take a while to compile pyds)..."
sudo docker-compose build

if [ $? -eq 0 ]; then
    echo "[INFO] Starting Application..."
    sudo docker-compose up
else
    echo "[ERROR] Build failed."
    exit 1
fi
