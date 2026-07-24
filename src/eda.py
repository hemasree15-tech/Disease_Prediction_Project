"""
eda.py

EDA means "Exploratory Data Analysis". This file creates graphs to
help us understand the dataset visually before building models.
All graphs are saved inside the graphs folder.
"""

import matplotlib
matplotlib.use("Agg")  # allows saving graphs without opening a window
import matplotlib.pyplot as plt
import numpy as np

from utils import GRAPH_FOLDER, make_folder_if_missing, print_heading


def plot_disease_distribution(data):
    """Bar graph showing how many patients belong to each disease."""
    plt.figure(figsize=(12, 6))
    data["Disease"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Disease Distribution")
    plt.xlabel("Disease")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/disease_distribution.png")
    plt.close()


def plot_symptom_frequency(data):
    """Bar graph showing how often each symptom appears in the data."""
    symptom_columns = [col for col in data.columns if col not in ["Disease", "Disease_encoded"]]
    symptom_sum = data[symptom_columns].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    symptom_sum.plot(kind="bar", color="orange")
    plt.title("Symptom Frequency")
    plt.xlabel("Symptom")
    plt.ylabel("Number of Occurrences")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/symptom_frequency.png")
    plt.close()


def plot_correlation_heatmap(data):
    """Heatmap showing correlation between symptom columns (using matplotlib only)."""
    symptom_columns = [col for col in data.columns if col not in ["Disease", "Disease_encoded"]]
    corr_matrix = data[symptom_columns].corr()

    plt.figure(figsize=(14, 10))
    plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(symptom_columns)), symptom_columns, rotation=90)
    plt.yticks(range(len(symptom_columns)), symptom_columns)
    plt.title("Correlation Heatmap of Symptoms")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/correlation_heatmap.png")
    plt.close()


def plot_missing_values(data):
    """Bar graph showing missing values in each column (should be 0 after cleaning)."""
    plt.figure(figsize=(12, 6))
    data.isnull().sum().plot(kind="bar", color="red")
    plt.title("Missing Values per Column")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/missing_values.png")
    plt.close()


def plot_histograms(data):
    """Histogram showing how many symptoms each patient has on average."""
    symptom_columns = [col for col in data.columns if col not in ["Disease", "Disease_encoded"]]
    symptom_count_per_patient = data[symptom_columns].sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(symptom_count_per_patient, bins=10, color="green", edgecolor="black")
    plt.title("Number of Symptoms per Patient")
    plt.xlabel("Symptom Count")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/histogram_symptom_count.png")
    plt.close()


def plot_boxplot(data):
    """Boxplot of total symptom count per patient."""
    symptom_columns = [col for col in data.columns if col not in ["Disease", "Disease_encoded"]]
    symptom_count_per_patient = data[symptom_columns].sum(axis=1)

    plt.figure(figsize=(6, 6))
    plt.boxplot(symptom_count_per_patient)
    plt.title("Boxplot of Symptom Count per Patient")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/boxplot_symptom_count.png")
    plt.close()


def plot_pie_chart(data):
    """Pie chart of the top 8 most common diseases."""
    top_diseases = data["Disease"].value_counts().head(8)

    plt.figure(figsize=(8, 8))
    plt.pie(top_diseases, labels=top_diseases.index, autopct="%1.1f%%")
    plt.title("Top 8 Diseases (Pie Chart)")
    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER + "/pie_chart_top_diseases.png")
    plt.close()


def run_eda(data):
    """Main function that calls every EDA graph function one by one."""
    print_heading("STEP 2: EXPLORATORY DATA ANALYSIS")
    make_folder_if_missing(GRAPH_FOLDER)

    plot_disease_distribution(data)
    plot_symptom_frequency(data)
    plot_correlation_heatmap(data)
    plot_missing_values(data)
    plot_histograms(data)
    plot_boxplot(data)
    plot_pie_chart(data)

    print("All EDA graphs saved inside the graphs folder.")
