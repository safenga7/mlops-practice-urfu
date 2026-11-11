#!/usr/bin/env python3
"""
Production Dataset Generator for Final Project
Exports the validated Iris dataset with strict schema and metadata.
"""

import os
import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "iris_dataset.csv")

def generate():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    
    # Rename columns to standard snake_case
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target"
    ]
    
    # Map target integer to class name string
    target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["target_name"] = df["target"].map(target_names)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated at {OUTPUT_FILE} with {len(df)} samples and {df.shape[1]} columns.")

if __name__ == "__main__":
    generate()
