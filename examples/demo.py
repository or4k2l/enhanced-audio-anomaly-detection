import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load dataset

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found!")
    return pd.read_csv(file_path)

# Function to split dataset

def split_dataset(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

# Main function to run the example

def main():
    # Load dataset
    dataset_path = 'path/to/your/dataset.csv'
    data = load_dataset(dataset_path)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(data)

    # Do something with the loaded and split data...
    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

if __name__ == '__main__':
    main()