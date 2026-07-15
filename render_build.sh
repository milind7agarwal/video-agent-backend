#!/usr/bin/env bash
# exit on error
set -o errexit

# Install python dependencies from Requirements.txt
pip install -r Requirements.txt

# Create a local bin folder in your root directory
mkdir -p bin

# Download and extract static FFmpeg if it's not already cached/present
if [ ! -f "bin/ffmpeg" ]; then
    echo "Downloading static FFmpeg for Render..."
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar -xf ffmpeg-release-amd64-static.tar.xz
    
    # Move binaries to the bin/ directory
    mv ffmpeg-*-static/ffmpeg bin/
    mv ffmpeg-*-static/ffprobe bin/
    
    # Clean up installation archives
    rm -rf ffmpeg-*-static*
    chmod +x bin/ffmpeg bin/ffprobe
    echo "FFmpeg installed successfully in local bin!"
fi