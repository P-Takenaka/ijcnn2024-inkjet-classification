# Preparations

## 1. Prepare Dataset
1. Download the dataset files from https://cloud.mi.hdm-stuttgart.de/s/mitfQm6DiPgtqCZ into the root folder
2. Combine the zip archive into a single file and unpack
```
zip -F dataset.zip --out combined_dataset.zip
mkdir -p images
unzip -d images combined_dataset.zip
```

## 2. Create conda environment

```
conda env create --file environment.yml
conda activate ijcnn-inkjet
```

## 3. Create dataset
We do the feature extraction as a preprocessing step before training a model, resulting in a new dataset file that only contains the extracted features:


```
python3 create_dataset.py 
```

This creates a file `extracted_features.pkl` in the current directory.

## 2. Train model

```
python3 train.py --config=src/configs/mlp_base.py
```

Performance on val set: per-crop performance
Performance on test set: per-document performance

# File Info

## metadata.pkl

Contains metadata to identify the individual images

## .env

Login data to mlflow. needs to be in same directory as the training script

## classifier.py

Sample training script

## create_dataset.py

Script for creating dataset

## feature_extraction.py

Script containing the feature extraction methods

## environment.yml

Conda environment file



