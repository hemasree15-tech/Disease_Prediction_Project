"""
prediction.py

This file lets the user enter a patient's details one by one and then
predicts whether the patient is likely to have heart disease using the
trained best model. It also saves every prediction made into a CSV file.
"""

import os

import pandas as pd

from data_preprocessing import encode_features
from feature_engineering import add_engineered_features
from utils import OUTPUT_FOLDER, make_folder_if_missing, print_heading


# Each numeric question: (column name, question text, minimum, maximum, number type)
NUMERIC_QUESTIONS = {
    "Age": ("Age in years", 1, 120, int),
    "RestingBP": ("Resting blood pressure (mm Hg)", 50, 250, int),
    "Cholesterol": ("Serum cholesterol (mg/dl)", 50, 700, int),
    "MaxHR": ("Maximum heart rate achieved", 50, 250, int),
    "Oldpeak": ("ST depression (Oldpeak), e.g. 0, 1.5", -3, 10, float),
}

# Each choice question: column name -> (question text, {typed answer: dataset value})
CHOICE_QUESTIONS = {
    "Sex": ("Sex", {"m": "M", "f": "F"}),
    "ChestPainType": (
        "Chest pain type (ATA = atypical angina, NAP = non-anginal pain, "
        "ASY = asymptomatic, TA = typical angina)",
        {"ata": "ATA", "nap": "NAP", "asy": "ASY", "ta": "TA"},
    ),
    "RestingECG": (
        "Resting ECG result (Normal, ST = ST-T abnormality, LVH = left ventricular hypertrophy)",
        {"normal": "Normal", "st": "ST", "lvh": "LVH"},
    ),
    "ExerciseAngina": ("Exercise-induced angina (Y/N)", {"y": "Y", "n": "N"}),
    "ST_Slope": (
        "Slope of the peak exercise ST segment (Up, Flat, Down)",
        {"up": "Up", "flat": "Flat", "down": "Down"},
    ),
}


def ask_number(question, minimum, maximum, number_type):
    """Keeps asking until the user types a valid number in the allowed range."""
    while True:
        text = input(question + " [" + str(minimum) + " to " + str(maximum) + "]: ").strip()
        try:
            value = number_type(text)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if minimum <= value <= maximum:
            return value
        print("Value must be between", minimum, "and", maximum)


def ask_choice(question, options):
    """Keeps asking until the user types one of the allowed options."""
    while True:
        text = input(question + " (" + "/".join(options.keys()) + "): ").strip().lower()
        if text in options:
            return options[text]
        print("Please type one of:", ", ".join(options.keys()))


def ask_yes_no(question):
    """Asks a Yes/No question and returns 1 (Yes) or 0 (No)."""
    while True:
        text = input(question + " (Yes/No): ").strip().lower()
        if text in ["yes", "y"]:
            return 1
        if text in ["no", "n"]:
            return 0
        print("Please type Yes or No.")


def ask_patient_questions():
    """
    Asks the user for every patient detail used by the model and
    returns them as a dictionary (same column names as heart.csv).
    """
    print("\nEnter patient details.")
    answers = {}

    age_text, low, high, kind = NUMERIC_QUESTIONS["Age"]
    answers["Age"] = ask_number(age_text, low, high, kind)
    answers["Sex"] = ask_choice(*CHOICE_QUESTIONS["Sex"])
    answers["ChestPainType"] = ask_choice(*CHOICE_QUESTIONS["ChestPainType"])

    for col in ["RestingBP", "Cholesterol"]:
        text, low, high, kind = NUMERIC_QUESTIONS[col]
        answers[col] = ask_number(text, low, high, kind)

    answers["FastingBS"] = ask_yes_no("Is fasting blood sugar greater than 120 mg/dl?")
    answers["RestingECG"] = ask_choice(*CHOICE_QUESTIONS["RestingECG"])

    text, low, high, kind = NUMERIC_QUESTIONS["MaxHR"]
    answers["MaxHR"] = ask_number(text, low, high, kind)

    answers["ExerciseAngina"] = ask_choice(*CHOICE_QUESTIONS["ExerciseAngina"])

    text, low, high, kind = NUMERIC_QUESTIONS["Oldpeak"]
    answers["Oldpeak"] = ask_number(text, low, high, kind)

    answers["ST_Slope"] = ask_choice(*CHOICE_QUESTIONS["ST_Slope"])

    return answers


def build_model_input(answers, feature_columns):
    """
    Converts the typed answers into the exact same format the model was
    trained on: one-hot encode the text columns, add MaxHR_percent, then
    arrange the columns in the training order (missing columns = 0).
    """
    raw_row = pd.DataFrame([answers])
    encoded_row = encode_features(raw_row)
    encoded_row = add_engineered_features(encoded_row)
    return encoded_row.reindex(columns=feature_columns, fill_value=0)


def predict_disease(best_model, scaler, label_encoder, feature_columns):
    """
    Main prediction loop. Keeps asking for patient details and
    predicting until the user chooses to stop.
    """
    print_heading("STEP 5: HEART DISEASE PREDICTION SYSTEM")

    make_folder_if_missing(OUTPUT_FOLDER)
    predictions_path = OUTPUT_FOLDER + "/predictions.csv"
    all_predictions = []

    while True:
        print("\n==============================")
        print("Heart Disease Prediction System")
        print("==============================")

        answers = ask_patient_questions()

        input_row = build_model_input(answers, feature_columns)
        input_scaled = scaler.transform(input_row)

        # Predict and get the confidence (probability) of the prediction
        prediction_encoded = best_model.predict(input_scaled)[0]
        predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

        try:
            probabilities = best_model.predict_proba(input_scaled)[0]
            confidence = round(max(probabilities) * 100, 2)
        except AttributeError:
            # Some models may not support predict_proba
            confidence = "Not Available"

        print("\n==============================")
        print("Prediction Result")
        print(predicted_label)
        print("\nPrediction Confidence")
        print(str(confidence) + "%")
        print("\nRecommended Action")
        print("Please consult a qualified doctor for confirmation.")
        print("This prediction is for educational purposes only.")
        print("==============================")

        # Save this prediction record
        record = answers.copy()
        record["Predicted_Result"] = predicted_label
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
    print("Thank you for using the Heart Disease Prediction System.")
