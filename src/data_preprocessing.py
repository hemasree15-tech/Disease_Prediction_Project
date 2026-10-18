"""
data_preprocessing.py

This file loads the dataset and cleans it so that it is ready to be
used for machine learning. It also splits the data into training and
testing parts.

Dataset: Heart Failure Prediction dataset (heart.csv)
Each row is one patient. The last column 'HeartDisease' is the target
(1 = patient has heart disease, 0 = patient is healthy).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    ENCODED_LABEL_COLUMN,
    LABEL_COLUMN,
    LABEL_NAMES,
    OUTPUT_FOLDER,
    TARGET_COLUMN,
    ZERO_MEANS_MISSING,
    make_folder_if_missing,
    print_heading,
)


def load_dataset():
    """
    Loads the CSV dataset from the dataset folder using a relative
    path, so it works on any computer.
    """
    data = pd.read_csv(DATASET_PATH)
    return data


def explore_dataset(data):
    """
    Prints basic information about the dataset so we can understand
    it before doing any machine learning.
    """
    print_heading("Dataset Shape")
    print(data.shape)

    print_heading("First 10 Rows")
    print(data.head(10))

    print_heading("Dataset Info")
    print(data.info())

    print_heading("Missing Values in Each Column")
    print(data.isnull().sum())

    print_heading("Impossible Zero Values (really missing data)")
    for col in ZERO_MEANS_MISSING:
        print(col, ":", int((data[col] == 0).sum()), "rows have 0")

    print_heading("Target Balance (HeartDisease: 1 = disease, 0 = healthy)")
    print(data[TARGET_COLUMN].value_counts())


def clean_dataset(data):
    """
    Cleans the dataset:
    1. Removes duplicate rows.
    2. Drops rows where the target is missing.
    3. Cholesterol = 0 and RestingBP = 0 are impossible for a living
       patient, so they are really missing values. We replace them
       with the median of the valid values.
    4. Any other missing numbers are filled with the median and any
       other missing text values with the most common value.
    """
    data = data.drop_duplicates().copy()
    data = data.dropna(subset=[TARGET_COLUMN])

    # 0 is not a real value for these columns -> treat as missing -> median
    for col in ZERO_MEANS_MISSING:
        data[col] = data[col].replace(0, np.nan)

    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == "object" or str(data[col].dtype).startswith("str"):
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())

    return data.reset_index(drop=True)


def encode_target(data):
    """
    Creates a readable text label (Healthy / Heart Disease) from the
    HeartDisease column and converts it into numbers using
    LabelEncoder (Healthy = 0, Heart Disease = 1).
    Returns the data and the encoder (needed later to convert numbers
    back into names).
    """
    data[LABEL_COLUMN] = data[TARGET_COLUMN].map(LABEL_NAMES)

    label_encoder = LabelEncoder()
    data[ENCODED_LABEL_COLUMN] = label_encoder.fit_transform(data[LABEL_COLUMN])
    return data, label_encoder


def encode_features(raw_features):
    """
    One-hot encodes the text columns (Sex, ChestPainType, RestingECG,
    ExerciseAngina, ST_Slope) because machine learning models need
    numbers. For example 'Sex' becomes two columns: Sex_F and Sex_M.

    This function is also used in prediction.py so that a new patient
    is encoded in exactly the same way as the training data.
    """
    return pd.get_dummies(raw_features, columns=CATEGORICAL_COLUMNS, dtype=int)


def split_features_and_target(data):
    """
    Separates the dataset into:
    x = input features (patient details, text columns one-hot encoded)
    y = target column (0 = healthy, 1 = heart disease)
    """
    raw_columns = [
        col for col in data.columns
        if col not in [TARGET_COLUMN, LABEL_COLUMN, ENCODED_LABEL_COLUMN]
    ]
    x = encode_features(data[raw_columns])
    y = data[ENCODED_LABEL_COLUMN]
    return x, y, list(x.columns)


def preprocess_data():
    """
    Main function of this file. It runs every preprocessing step in
    order and returns everything that is needed for training later.
    """
    print_heading("STEP 1: DATA PREPROCESSING")

    data = load_dataset()
    explore_dataset(data)

    data = clean_dataset(data)
    print_heading("Dataset Shape After Cleaning")
    print(data.shape)

    data, label_encoder = encode_target(data)
    x, y, feature_columns = split_features_and_target(data)

    # Train-Test split (80% training, 20% testing).
    # stratify=y keeps the same healthy / disease ratio in both parts.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save the cleaned dataset to the output folder
    make_folder_if_missing(OUTPUT_FOLDER)
    processed_path = OUTPUT_FOLDER + "/processed_dataset.csv"
    data.to_csv(processed_path, index=False)
    print("\nProcessed dataset saved at:", processed_path)

    return x_train, x_test, y_train, y_test, feature_columns, label_encoder, data
