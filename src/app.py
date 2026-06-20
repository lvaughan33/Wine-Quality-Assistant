import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai

from src.features import FEATURES

# Load .env from the project root directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", api_key is not None)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check that your .env file exists "
        "in the project root and contains GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=api_key)

# Load the trained model
model = joblib.load("models/best_model.pkl")

def extract_features(user_input: str) -> dict:
    prompt = f"""
You extract wine characteristics from user messages.

Return ONLY valid JSON.

Required features:
{FEATURES}

Rules:
- Use the exact feature names shown above.
- Use numeric values only.
- Use null for missing features.
- Do not include any extra fields.

User message:
{user_input}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    response_text = response.text.strip()

    # Remove markdown formatting if present
    response_text = (
        response_text.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(response_text)

def predict_quality(features: dict):
    missing = [
        feature
        for feature in FEATURES
        if features.get(feature) is None
    ]

    if missing:
        return None, missing

    ordered_features = {
        feature: features[feature]
        for feature in FEATURES
    }

    input_df = pd.DataFrame([ordered_features])

    prediction = model.predict(input_df)[0]

    return prediction, []

def generate_response(prediction: float) -> str:
    prompt = f"""
A machine learning model predicted a wine quality score of {prediction:.1f}
on a scale from 0 to 10.

Write a short, friendly explanation for a user.

Keep the response to 2-3 sentences.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()

def main():
    print("\nWine Quality Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Describe your wine: ")

        if user_input.lower() == "quit":
            break

        try:
            features = extract_features(user_input)

            prediction, missing = predict_quality(features)

            if missing:
                print("\nI still need these values:")

                for feature in missing:
                    print(f"- {feature}")

                print()
                continue

            print(f"\nPredicted wine quality score: {prediction:.1f}")

            explanation = generate_response(prediction)

            print(f"\n{explanation}\n")

        except json.JSONDecodeError:
            print(
                "\nI couldn't understand the wine characteristics. "
                "Please try again.\n"
            )

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()