#!/bin/bash

set -e  # Exit on error

# =========================
# Base directories (relative)
# =========================
BASE_DIR="./datasets"
DOWNLOAD_DIR="./downloads"

mkdir -p "$BASE_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo " Base directory: $BASE_DIR"
echo " Download directory: $DOWNLOAD_DIR"
echo "-----------------------------------"

# =========================
# Helper function
# =========================
download_and_extract () {
  DATASET_NAME=$1
  KAGGLE_URL=$2
  ZIP_NAME=$3
  TARGET_DIR=$4

  if [[ -z "$KAGGLE_URL" || "$KAGGLE_URL" == *"<"* ]]; then
    echo " URL for $DATASET_NAME not configured. Please set the dataset URL or place files in $BASE_DIR/$TARGET_DIR"
    echo "-----------------------------------"
    return 0
  fi

  echo "  Downloading $DATASET_NAME..."
  curl -L -o "$DOWNLOAD_DIR/$ZIP_NAME" "$KAGGLE_URL"

  echo "  Extracting $DATASET_NAME..."
  mkdir -p "$BASE_DIR/$TARGET_DIR"
  unzip -q "$DOWNLOAD_DIR/$ZIP_NAME" -d "$BASE_DIR/$TARGET_DIR"

  echo " Cleaning up zip file..."
  rm "$DOWNLOAD_DIR/$ZIP_NAME"

  echo " $DATASET_NAME ready!"
  echo "-----------------------------------"
}

# =========================
# Downloads
# Provide your dataset download URLs below:
# =========================

download_and_extract \
  "Skin Disease Dataset" \
  "<SKIN_DISEASE_DATASET_URL>" \
  "skin_disease.zip" \
  "skin_disease"

download_and_extract \
  "Nail Disease Dataset" \
  "<NAIL_DISEASE_DATASET_URL>" \
  "nail_disease.zip" \
  "nail_disease"

download_and_extract \
  "Eye Disease Dataset" \
  "<EYE_DISEASE_DATASET_URL>" \
  "eye_disease.zip" \
  "eye_disease"

download_and_extract \
  "Alzheimer's Dataset" \
  "<ALZHEIMERS_DATASET_URL>" \
  "alzheimers.zip" \
  "alzheimers"

echo   All datasets downloaded and organized successfully!"
