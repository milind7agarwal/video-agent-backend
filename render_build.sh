#!/usr/bin/env bash
# exit on error
set -o errexit
 
echo "Installing lightweight Python dependencies..."
pip install -r requirements.render.txt
 
echo "Downloading static ffmpeg binary for Render..."
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
rm ffmpeg-release-amd64-static.tar.xz
FFMPEG_DIR=$(find . -maxdepth 1 -type d -name "ffmpeg-*-static" | head -n 1)
mkdir -p $HOME/.local/bin
mv $FFMPEG_DIR/ffmpeg $HOME/.local/bin/
mv $FFMPEG_DIR/ffprobe $HOME/.local/bin/
rm -rf $FFMPEG_DIR
 
echo "Build complete. FFmpeg installed."
 