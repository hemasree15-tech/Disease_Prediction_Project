"""
train_models.py

This file trains several different machine learning models on the
data, compares their accuracy, and picks the best one automatically.
"""

import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import MODEL_FOLDER, OUTPUT_FOLDER, make_folder_if_missing, print_heading


def get_all_models():
    """
    Returns a dictionary of all the machine learning models we want
    to train and compare. Using a dictionary makes it easy to loop
    through every model with a simple 'for' loop.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    return models


def train_and_compare_models(x_train, x_test, y_train, y_test):
    """
    Trains every model, checks its accuracy on the test data, and
    stores all the trained models and their accuracy scores.
    """
    print_heading("STEP 3: TRAINING AND COMPARING MACHINE LEARNING MODELS")

    # Scale the features (helps models like SVM, KNN and Logistic Regression)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = get_all_models()
    trained_models = {}
    accuracy_scores = {}

    for model_name, model in models.items():
        print("\nTraining:", model_name)
        model.fit(x_train_scaled, y_train)
        accuracy = model.score(x_test_scaled, y_test)
        accuracy_scores[model_name] = round(accuracy * 100, 2)
        trained_models[model_name] = model
        print(model_name, "Accuracy:", accuracy_scores[model_name], "%")

    # Create accuracy comparison table
    accuracy_table = pd.DataFrame(
        list(accuracy_scores.items()), columns=["Model", "Accuracy (%)"]
    )
    accuracy_table = accuracy_table.sort_values(by="Accuracy (%)", ascending=False)

    print_heading("Accuracy Comparison Table")
    print(accuracy_table)

    # Save the accuracy table to the output folder
    make_folder_if_missing(OUTPUT_FOLDER)
    accuracy_table.to_csv(OUTPUT_FOLDER + "/accuracy_results.csv", index=False)

    # Automatically pick the best model (highest accuracy)
    best_model_name = accuracy_table.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print("\nBest Model Selected:", best_model_name)

    # Save the best model and the scaler using pickle
    make_folder_if_missing(MODEL_FOLDER)
    with open(MODEL_FOLDER + "/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_FOLDER + "/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Best model and scaler saved inside the models folder.")

    return best_model, best_model_name, scaler, x_test_scaled, accuracy_table
