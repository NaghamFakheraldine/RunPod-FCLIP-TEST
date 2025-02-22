#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Preload the FashionCLIP model
echo "Preloading FashionCLIP model"
python3 -c "from fashion_clip.fashion_clip import FashionCLIP; FashionCLIP('fashion-clip')"

# Start the Runpod worker
echo "Starting Runpod worker"
python3 fclip.py