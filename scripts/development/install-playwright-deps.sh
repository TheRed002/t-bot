#!/bin/bash
# Install Playwright system dependencies

echo "Installing Playwright system dependencies..."
echo "This script needs sudo access to install system packages."
echo ""

# Update package list
sudo apt-get update

# Install dependencies for Playwright browsers
sudo apt-get install -y \
    libgtk-4-1 \
    libgraphene-1.0-0 \
    libxslt1.1 \
    libwoff1 \
    libvpx7 \
    libopus0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-gl1.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    libflite1 \
    libwebpdemux2 \
    libavif13 \
    libharfbuzz-icu0 \
    libenchant-2-2 \
    libsecret-1-0 \
    libhyphen0 \
    libmanette-0.2-0 \
    libx264-163 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2

echo ""
echo "✅ Playwright system dependencies installed!"
echo ""
echo "You can now use Playwright for browser automation."