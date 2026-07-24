"""
feature_engineering.py

Feature engineering means creating new useful columns (features) from
the existing data to help the machine learning model learn better.

For this symptom dataset, the most useful extra feature is the total
number of symptoms a patient has. This can help models understand how
severe a case might be.
"""

from utils import print_heading


def add_symptom_count_feature(x_train, x_test):
    """
    Adds a new column called 'total_symptoms' which is simply the sum
    of all symptom columns (how many symptoms = 1 for that patient).
    This is added to both the training and testing data.
    """
    x_train = x_train.copy()
    x_test = x_test.copy()

    x_train["total_symptoms"] = x_train.sum(axis=1)
    x_test["total_symptoms"] = x_test.sum(axis=1)

    return x_train, x_test


def run_feature_engineering(x_train, x_test):
    """Main function of this file. Runs all feature engineering steps."""
    print_heading("Feature Engineering: Adding total_symptoms column")
    x_train, x_test = add_symptom_count_feature(x_train, x_test)
    print("New feature 'total_symptoms' added successfully.")
    return x_train, x_test
