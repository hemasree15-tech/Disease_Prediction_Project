"""
evaluation.py

This file checks how good the best model really is, using standard
evaluation metrics like accuracy, precision, recall and F1 score.
It also saves the confusion matrix as a picture.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils import GRAPH_FOLDER, OUTPUT_FOLDER, make_folder_if_missing, print_heading


def evaluate_model(best_model, x_test_scaled, y_test, label_encoder):
    """
    Calculates evaluation metrics for the best model and saves a
    text report plus a confusion matrix graph.
    """
    print_heading("STEP 4: MODEL EVALUATION")

    y_pred = best_model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print_heading("Classification Report")
    print(report)

    # Save results to a text file inside the output folder
    make_folder_if_missing(OUTPUT_FOLDER)
    with open(OUTPUT_FOLDER + "/evaluation_report.txt", "w") as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write("========================\n")
        f.write("Accuracy : " + str(round(accuracy * 100, 2)) + "%\n")
        f.write("Precision: " + str(round(precision * 100, 2)) + "%\n")
        f.write("Recall   : " + str(round(recall * 100, 2)) + "%\n")
        f.write("F1 Score : " + str(round(f1 * 100, 2)) + "%\n\n")
        f.write("Classification Report\n")
        f.write(report)

    # Save confusion matrix as an image
    make_folder_if_missing(GRAPH_FOLDER)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/confusion_matrix.png")
    plt.close()

    print("\nEvaluation report and confusion matrix graph saved successfully.")

    return accuracy, precision, recall, f1
