"""
data_preprocessing.py

This file loads the dataset and cleans it so that it is ready to be
used for machine learning. It also splits the data into training and
testing parts.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import DATASET_PATH, OUTPUT_FOLDER, make_folder_if_missing, print_heading


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


def clean_dataset(data):
    """
    Removes duplicate rows and fills missing values.
    For symptom columns (numbers), missing values are filled with 0.
    For the Disease column (text), missing rows are simply dropped.
    """
    # Remove duplicate rows
    data = data.drop_duplicates()

    # Fill missing values in symptom columns with 0 (means symptom not present)
    symptom_columns = [col for col in data.columns if col != "Disease"]
    data[symptom_columns] = data[symptom_columns].fillna(0)

    # Drop rows where the Disease (target) value itself is missing
    data = data.dropna(subset=["Disease"])

    return data


def encode_target(data):
    """
    Converts the Disease column (text) into numbers using
    LabelEncoder, because machine learning models need numeric data.
    Returns the encoded data and the encoder object (needed later to
    convert numbers back into disease names).
    """
    label_encoder = LabelEncoder()
    data["Disease_encoded"] = label_encoder.fit_transform(data["Disease"])
    return data, label_encoder


def split_features_and_target(data):
    """
    Separates the dataset into:
    x = input features (symptom columns)
    y = target column (encoded disease)
    """
    feature_columns = [col for col in data.columns if col not in ["Disease", "Disease_encoded"]]
    x = data[feature_columns]
    y = data["Disease_encoded"]
    return x, y, feature_columns


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

    # Train-Test split (80% training, 20% testing)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    # Save the cleaned dataset to the output folder
    make_folder_if_missing(OUTPUT_FOLDER)
    processed_path = OUTPUT_FOLDER + "/processed_dataset.csv"
    data.to_csv(processed_path, index=False)
    print("\nProcessed dataset saved at:", processed_path)

    return x_train, x_test, y_train, y_test, feature_columns, label_encoder, data
