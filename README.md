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

# Reference
If you make use of this code, please cite the following publication:
```
@INPROCEEDINGS{Take2406:Classification,
AUTHOR="Patrick Takenaka and Manuel Eberhardinger and Daniel {Grie{\ss}haber} and
Johannes Maucher",
TITLE="Classification of Inkjet Printers Based on Droplet Statistics",
BOOKTITLE="2024 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2024)",
ADDRESS="Yokohama, Japan",
PAGES="6.61",
DAYS=28,
MONTH=jun,
YEAR=2024,
KEYWORDS="printer classification; frequency domain features; feature extraction;
feature engineering",
ABSTRACT="Knowing the printer model used to print a given document may provide a
crucial lead towards identifying counterfeits or conversely verifying the
validity of a real document. Inkjet printers produce probabilistic droplet
patterns that appear to be distinct for each printer model and as such we
investigate the utilization of droplet characteristics including frequency
domain features extracted from printed document scans for the
classification of the underlying printer model. We collect and publish a
dataset of high resolution document scans and show that our extracted
features are informative enough to enable a neural network to distinguish
not only the printer manufacturer, but also individual printer models."
}


```

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



