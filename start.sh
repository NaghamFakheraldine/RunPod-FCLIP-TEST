#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Start the Runpod worker
echo "Starting Runpod worker"
python3 -c "import runpod; from fclip import handler; runpod.serverless.start({'handler': handler})"
