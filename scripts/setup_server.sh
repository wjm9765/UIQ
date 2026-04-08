#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo "   Setting up GPU Server Environment...      "
echo "============================================="

# 1. Update package list and install system dependencies
echo "[1/3] Installing system dependencies via apt..."
sudo apt-get update -y
# ffmpeg is strictly required for yt-dlp memory streaming
sudo apt-get install -y ffmpeg curl

# 2. Check and install 'uv' if it doesn't exist
echo "[2/3] Checking 'uv' package manager..."
if ! command -v uv &> /dev/null
then
    echo "'uv' is not installed. Installing 'uv' via pip..."
    pip install uv
else
    echo "'uv' is already installed."
fi

# 3. Install Python dependencies using uv
echo "[3/3] Syncing Python dependencies..."
uv sync

echo "============================================="
echo " 🎉 Environment setup completed successfully! "
echo "============================================="
echo "Next steps:"
echo "  1. run ./scripts/setup_vggsound.py"
echo "  2. export HF_TOKEN='your_huggingface_token'"
echo "  3. run ./scripts/setup_models.py"
echo "  4. run ./scripts/run_eval.py"
