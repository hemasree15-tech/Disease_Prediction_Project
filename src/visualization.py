"""
visualization.py

This file creates extra graphs after model training: the accuracy
comparison chart and the feature importance graph.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import GRAPH_FOLDER, make_folder_if_missing, print_heading


def plot_accuracy_comparison(accuracy_table):
    """Bar graph comparing the accuracy of all trained models."""
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_table["Model"], accuracy_table["Accuracy (%)"], color="purple")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/accuracy_comparison.png")
    plt.close()


def plot_feature_importance(best_model, feature_columns):
    """
    Bar graph showing which symptoms are most important for
    prediction. This only works for tree-based models
    (Random Forest, Decision Tree, Gradient Boosting) which have a
    'feature_importances_' attribute.
    """
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # top 15 features

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importances[indices], color="teal")
        plt.xticks(
            range(len(indices)),
            [feature_columns[i] for i in indices],
            rotation=90,
        )
        plt.title("Top 15 Important Symptoms")
        plt.tight_layout()
        plt.savefig(GRAPH_FOLDER + "/feature_importance.png")
        plt.close()
        print("Feature importance graph saved.")
    else:
        print("Selected best model does not support feature importance graph.")


def plot_disease_count(data):
    """A simple horizontal bar graph of disease counts."""
    plt.figure(figsize=(10, 8))
    data["Disease"].value_counts().plot(kind="barh", color="brown")
    plt.title("Disease Count Graph")
    plt.xlabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/disease_count.png")
    plt.close()


def run_visualizations(accuracy_table, best_model, feature_columns, data):
    """Main function that calls every extra visualization function."""
    print_heading("STEP 6: ADDITIONAL VISUALIZATIONS")
    make_folder_if_missing(GRAPH_FOLDER)

    plot_accuracy_comparison(accuracy_table)
    plot_feature_importance(best_model, feature_columns)
    plot_disease_count(data)

    print("All additional graphs saved inside the graphs folder.")
