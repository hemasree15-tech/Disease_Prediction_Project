"""
feature_engineering.py

Feature engineering means creating new useful columns (features) from
the existing data to help the machine learning model learn better.

For this heart dataset, a useful extra feature is how close the
patient's maximum heart rate (MaxHR) is to the maximum expected for
their age (220 - Age). Doctors use this "percent of age-predicted
maximum heart rate" to judge how well the heart responds to exercise.
"""

from utils import print_heading

NEW_FEATURE_NAME = "MaxHR_percent"


def add_engineered_features(df):
    """
    Adds the new column 'MaxHR_percent' to any table that has the
    columns 'Age' and 'MaxHR'. It is used for the training data here
    and for a new patient in prediction.py.
    """
    df = df.copy()
    df[NEW_FEATURE_NAME] = (df["MaxHR"] / (220 - df["Age"])) * 100
    return df


def run_feature_engineering(x_train, x_test):
    """Main function of this file. Runs all feature engineering steps."""
    print_heading("Feature Engineering: Adding MaxHR_percent column")
    x_train = add_engineered_features(x_train)
    x_test = add_engineered_features(x_test)
    print("New feature '" + NEW_FEATURE_NAME + "' added successfully.")
    return x_train, x_test
