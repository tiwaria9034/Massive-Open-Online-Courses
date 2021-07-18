"""
Module to implement IO-related utility functions.
"""

import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


def load_mnist(data_dir, mode="train"):
    """Load MNIST dataset."""
    # Glob all csv files and find the one corresponding to mode of operation.
    csv_file = Path(data_dir, f"{mode}.csv")

    # Load data if the csv file exists in data directory
    if csv_file.exists():
        # Load data
        df = pd.read_csv(csv_file)
        data = df.values

        # Separate pixel values and labels
        x_data = data[:, 1:] / 255.0
        y_data = data[:, 0]

        # Shuffle
        x_data, y_data = shuffle(x_data, y_data)

        return x_data, y_data
