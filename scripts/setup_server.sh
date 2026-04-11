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
sudo apt-get install -y ffmpeg curl python3-pip python3-venv

# 2. Check and install 'uv' if it doesn't exist
echo "[2/3] Checking 'uv' package manager..."

# Ensure pip is installed before installing uv
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null
then
    echo "'pip' is not installed. Installing pip..."
    sudo apt-get install -y python3-pip
fi

if ! command -v uv &> /dev/null
then
    echo "'uv' is not installed. Installing 'uv' via pip..."
    pip3 install uv || pip install uv
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
echo "  1. export HF_TOKEN='your_huggingface_token'"
echo "  2. python scripts/setup_models.py"
echo "  3. python scripts/evaluate_all_claps.py"
