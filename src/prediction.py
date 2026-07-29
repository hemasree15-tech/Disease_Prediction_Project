"""
prediction.py

This file lets the user enter their symptoms one by one (Yes/No) and
then predicts the most likely disease using the trained best model.
It also saves every prediction made into a CSV file.
"""

import os

import numpy as np
import pandas as pd

from utils import OUTPUT_FOLDER, make_folder_if_missing, print_heading


def ask_symptom_questions(feature_columns):
    """
    Asks the user a Yes/No question for every symptom column and
    converts the answer into 1 (Yes) or 0 (No).
    The 'total_symptoms' feature (if present) is calculated
    automatically, not asked as a question.
    """
    answers = {}
    print("\nEnter patient details.")

    for symptom in feature_columns:
        if symptom == "total_symptoms":
            continue  # this is calculated automatically, not asked

        # Make the question readable, e.g. "high_fever" -> "high fever"
        question_text = symptom.replace("_", " ")
        while True:
            user_input = input("Do you have " + question_text + "? (Yes/No): ").strip().lower()
            if user_input in ["yes", "y"]:
                answers[symptom] = 1
                break
            elif user_input in ["no", "n"]:
                answers[symptom] = 0
                break
            else:
                print("Please type Yes or No.")

    # Add total_symptoms if it is expected by the model
    if "total_symptoms" in feature_columns:
        answers["total_symptoms"] = sum(answers.values())

    return answers


def predict_disease(best_model, scaler, label_encoder, feature_columns):
    """
    Main prediction loop. Keeps asking the user for symptoms and
    predicting diseases until the user chooses to stop.
    """
    print_heading("STEP 5: DISEASE PREDICTION SYSTEM")

    make_folder_if_missing(OUTPUT_FOLDER)
    predictions_path = OUTPUT_FOLDER + "/predictions.csv"
    all_predictions = []

    while True:
        print("\n==============================")
        print("Disease Prediction System")
        print("==============================")

        answers = ask_symptom_questions(feature_columns)

        # Arrange the answers in the same column order as training data
        input_row = pd.DataFrame([answers])[feature_columns]
        input_scaled = scaler.transform(input_row)

        # Predict disease and get the confidence (probability) of the prediction
        prediction_encoded = best_model.predict(input_scaled)[0]
        predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]

        try:
            probabilities = best_model.predict_proba(input_scaled)[0]
            confidence = round(max(probabilities) * 100, 2)
        except AttributeError:
            # Some models may not support predict_proba
            confidence = "Not Available"

        print("\n==============================")
        print("Predicted Disease")
        print(predicted_disease)
        print("\nPrediction Confidence")
        print(str(confidence) + "%")
        print("\nRecommended Action")
        print("Please consult a qualified doctor for confirmation.")
        print("This prediction is for educational purposes only.")
        print("==============================")

        # Save this prediction record
        record = answers.copy()
        record["Predicted_Disease"] = predicted_disease
        record["Confidence(%)"] = confidence
        all_predictions.append(record)

        again = input("\nDo you want to make another prediction? (Yes/No): ").strip().lower()
        if again not in ["yes", "y"]:
            break

    # Save all predictions made in this session to a CSV file
    predictions_df = pd.DataFrame(all_predictions)
    if os.path.exists(predictions_path):
        old_predictions = pd.read_csv(predictions_path)
        predictions_df = pd.concat([old_predictions, predictions_df], ignore_index=True)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nAll predictions saved to:", predictions_path)
    print("Thank you for using the Disease Prediction System.")
