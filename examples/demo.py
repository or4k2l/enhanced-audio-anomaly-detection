# Updated examples/demo.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset loading
# Assuming there's a function defined to load the dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

# Load the dataset once
file_path = 'path/to/dataset.csv'
dataset = load_dataset(file_path)

# Define function to train and test models
def train_and_test_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Add model training and evaluation code here
    return model

# Assuming dataset has 'features' and 'labels'
features = dataset.drop('label', axis=1)
labels = dataset['label']

# Train and test the model
model = train_and_test_model(features, labels)