from pathlib import Path

import joblib
from sklearn.metrics import r2_score

from src.preprocess import load_data, split_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "winequality-red.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

def test_model_prediction_shape():
    df = load_data(DATA_PATH)

    _, X_test, _, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)

def test_model_meets_minimum_r2_threshold():
    df = load_data(DATA_PATH)

    _, X_test, _, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    assert score > 0.50