"""
main.py

This is the ONLY file you need to run.
It runs the complete project step by step:
1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training and Comparison
5. Model Evaluation
6. Additional Visualizations
7. Disease Prediction (interactive)

HOW TO RUN THIS PROJECT IN SPYDER:
1. Open Spyder (Anaconda).
2. Open this file (main.py) using File -> Open.
3. Make sure Spyder's working directory is set to this 'src' folder
   (Spyder usually does this automatically when you open a file).
4. Click the green "Run" button, or press F5.
5. Follow the instructions printed in the console (right side / IPython console).
"""

from utils import print_heading

from data_preprocessing import preprocess_data
from eda import run_eda
from feature_engineering import run_feature_engineering
from train_models import train_and_compare_models
from evaluation import evaluate_model
from visualization import run_visualizations
from prediction import predict_disease


def main():
    print_heading("MULTI-DISEASE PREDICTION SYSTEM USING MACHINE LEARNING")

    # Step 1: Data Preprocessing
    x_train, x_test, y_train, y_test, feature_columns, label_encoder, data = preprocess_data()

    # Step 2: Exploratory Data Analysis
    run_eda(data)

    # Step 3: Feature Engineering
    x_train, x_test = run_feature_engineering(x_train, x_test)
    feature_columns = list(x_train.columns)

    # Step 4: Train and Compare Machine Learning Models
    best_model, best_model_name, scaler, x_test_scaled, accuracy_table = train_and_compare_models(
        x_train, x_test, y_train, y_test
    )

    # Step 5: Model Evaluation
    evaluate_model(best_model, x_test_scaled, y_test, label_encoder)

    # Step 6: Additional Visualizations
    run_visualizations(accuracy_table, best_model, feature_columns, data)

    # Step 7: Disease Prediction (interactive, user enters symptoms)
    predict_disease(best_model, scaler, label_encoder, feature_columns)

    print_heading("PROJECT EXECUTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
