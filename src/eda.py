"""
eda.py

EDA means "Exploratory Data Analysis". This file creates graphs to
help us understand the heart disease dataset visually before building
models. All graphs are saved inside the graphs folder.
"""

import matplotlib
matplotlib.use("Agg")  # allows saving graphs without opening a window
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    CATEGORICAL_COLUMNS,
    GRAPH_FOLDER,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    make_folder_if_missing,
    print_heading,
)


def plot_disease_distribution(data):
    """Bar graph showing how many patients are healthy vs have heart disease."""
    plt.figure(figsize=(6, 5))
    counts = data[LABEL_COLUMN].value_counts()
    plt.bar(counts.index, counts.values, color=["skyblue", "salmon"])
    plt.title("Disease Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/disease_distribution.png")
    plt.close()


def plot_categorical_features(data):
    """For each text column, shows how many patients with and without disease fall in each category."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(CATEGORICAL_COLUMNS):
        table = data.groupby([col, TARGET_COLUMN]).size().unstack(fill_value=0)
        table.plot(kind="bar", ax=axes[i], color=["skyblue", "salmon"])
        axes[i].set_title(col + " vs Heart Disease")
        axes[i].set_ylabel("Patients")
        axes[i].legend(["Healthy", "Heart Disease"])
        axes[i].tick_params(axis="x", rotation=0)

    axes[-1].axis("off")  # 5 text columns, so the 6th box stays empty
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/categorical_features.png")
    plt.close()


def plot_correlation_heatmap(data):
    """Heatmap showing correlation between the numeric columns and the target (matplotlib only)."""
    columns = NUMERIC_COLUMNS + [TARGET_COLUMN]
    corr_matrix = data[columns].corr()

    plt.figure(figsize=(8, 7))
    plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)

    # write the correlation value inside every box
    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(j, i, round(corr_matrix.iloc[i, j], 2), ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/correlation_heatmap.png")
    plt.close()


def plot_missing_values(data):
    """Bar graph showing missing values in each column (should be 0 after cleaning)."""
    plt.figure(figsize=(10, 6))
    data.isnull().sum().plot(kind="bar", color="red")
    plt.title("Missing Values per Column")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/missing_values.png")
    plt.close()


def plot_histograms(data):
    """Histograms showing the spread of every numeric column."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLUMNS):
        axes[i].hist(data[col], bins=20, color="green", edgecolor="black")
        axes[i].set_title(col)
        axes[i].set_ylabel("Number of Patients")

    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/histograms_numeric.png")
    plt.close()


def plot_boxplot(data):
    """Boxplots comparing healthy patients with heart disease patients for the main numeric columns."""
    columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    fig, axes = plt.subplots(1, len(columns), figsize=(16, 5))

    for i, col in enumerate(columns):
        healthy = data[data[TARGET_COLUMN] == 0][col]
        disease = data[data[TARGET_COLUMN] == 1][col]
        axes[i].boxplot([healthy, disease], tick_labels=["Healthy", "Disease"])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/boxplot_numeric_by_disease.png")
    plt.close()


def plot_pie_chart(data):
    """Pie chart of the healthy vs heart disease patients."""
    counts = data[LABEL_COLUMN].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["salmon", "skyblue"])
    plt.title("Healthy vs Heart Disease (Pie Chart)")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/pie_chart_disease.png")
    plt.close()


def run_eda(data):
    """Main function that calls every EDA graph function one by one."""
    print_heading("STEP 2: EXPLORATORY DATA ANALYSIS")
    make_folder_if_missing(GRAPH_FOLDER)

    plot_disease_distribution(data)
    plot_categorical_features(data)
    plot_correlation_heatmap(data)
    plot_missing_values(data)
    plot_histograms(data)
    plot_boxplot(data)
    plot_pie_chart(data)

    print("All EDA graphs saved inside the graphs folder.")
