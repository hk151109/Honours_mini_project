# ============================================================
# DATASET DOWNLOAD SCRIPT
# ============================================================
# NOTE: The dataset in this folder is only a SAMPLE for demo.
# Run this script to download the FULL dataset from Kaggle.
#
# Dataset: Wildfire Prediction Dataset
# Source: https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
# Classes: wildfire, nowildfire
# Structure: train/, valid/, test/ splits
# ============================================================

import kagglehub

# Download latest version 
path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset") 
print("Path to dataset files:", path)
