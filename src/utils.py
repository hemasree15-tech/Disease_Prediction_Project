"""
utils.py

This file has small helper functions that are used by many other
files in this project. Keeping them here avoids repeating the same
code again and again (this is called "code reuse").
"""

import os


def make_folder_if_missing(folder_path):
    """
    Checks if a folder exists. If it does not exist, it creates it.
    This is used so the program never crashes with a
    'folder not found' error.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def print_heading(text):
    """
    Prints a heading with a line of '=' signs above and below it.
    This is just used to make the console output look neat.
    """
    line = "=" * 50
    print("\n" + line)
    print(text)
    print(line)


# These are the base folder paths used everywhere in the project.
# They are all relative paths, so the project works on any computer
# as long as the folder structure is not changed.
DATASET_PATH = os.path.join("..", "dataset", "disease_dataset.csv")
OUTPUT_FOLDER = os.path.join("..", "output")
GRAPH_FOLDER = os.path.join("..", "graphs")
MODEL_FOLDER = os.path.join("..", "models")
