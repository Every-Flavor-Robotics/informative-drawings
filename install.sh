#!/bin/bash

# Ensure the script runs from the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define the path to the model.zip file
ZIP_FILE="./checkpoints/model.zip"

# Check if the zip file exists
if [ -f "$ZIP_FILE" ]; then
  # Unzip the file into the checkpoints directory
  unzip -o "$ZIP_FILE" -d ./checkpoints
  # Move the unzipped files from ./checkpoints/model to the checkpoints directory
  mv ./checkpoints/model/* ./checkpoints
  # Remove the empty model directory
  rm -r ./checkpoints/model
  echo "Unzipped $ZIP_FILE into ./checkpoints"
else
  echo "Error: $ZIP_FILE does not exist."
  exit 1
fi