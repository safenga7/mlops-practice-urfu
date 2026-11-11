"""
Dataset preparation script for Lab 4 (DVC Versioning).
Downloads and prepares versions of the Titanic dataset:
  - Version 1: Filtered features: Pclass, Sex, Age
  - Version 2: Missing values in Age imputed with column mean
  - Version 3: One-hot encoding for the string feature 'Sex'
"""

import os
import sys
import argparse
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def download_raw_if_needed(raw_path: str) -> pd.DataFrame:
    """Downloads raw Titanic dataset if not cached."""
    if not os.path.exists(raw_path):
        print(f"Downloading raw dataset from {DATA_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(DATA_URL, raw_path)
    return pd.read_csv(raw_path)


def create_version_1(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Version 1: Extract Pclass, Sex, Age."""
    df_v1 = df_raw[["Pclass", "Sex", "Age"]].copy()
    print("Version 1 created: features ['Pclass', 'Sex', 'Age'].", file=sys.stderr)
    return df_v1


def create_version_2(df_v1: pd.DataFrame) -> pd.DataFrame:
    """Version 2: Impute missing Age with mean."""
    df_v2 = df_v1.copy()
    mean_age = df_v2["Age"].mean()
    nan_count_before = df_v2["Age"].isna().sum()
    df_v2["Age"] = df_v2["Age"].fillna(mean_age)
    nan_count_after = df_v2["Age"].isna().sum()
    print(
        f"Version 2 created: Age NaN imputed with mean ({mean_age:.2f}). "
        f"Missing before: {nan_count_before}, after: {nan_count_after}.",
        file=sys.stderr,
    )
    return df_v2


def create_version_3(df_v2: pd.DataFrame) -> pd.DataFrame:
    """Version 3: One-hot encode string feature Sex."""
    df_v3 = pd.get_dummies(df_v2, columns=["Sex"], dtype=int)
    print(
        f"Version 3 created: One-hot encoded 'Sex'. Columns: {df_v3.columns.tolist()}",
        file=sys.stderr,
    )
    return df_v3


def main():
    parser = argparse.ArgumentParser(description="Prepare Titanic dataset versions for DVC.")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Dataset version to generate (1, 2, or 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/titanic.csv",
        help="Path where the versioned dataset will be written",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "raw_titanic.csv")
    output_path = os.path.isabs(args.output) and args.output or os.path.join(base_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_raw = download_raw_if_needed(raw_path)

    v1 = create_version_1(df_raw)
    if args.version == 1:
        df_target = v1
    elif args.version == 2:
        df_target = create_version_2(v1)
    elif args.version == 3:
        v2 = create_version_2(v1)
        df_target = create_version_3(v2)

    df_target.to_csv(output_path, index=False)
    print(f"Dataset v{args.version} written to {output_path} ({len(df_target)} rows, {df_target.shape[1]} cols)")
    print(df_target.head(3))


if __name__ == "__main__":
    main()
