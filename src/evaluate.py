from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_data, split_data

def main():
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "raw" / "winequality-red.csv"
    model_path = project_root / "models" / "best_model.pkl"

    # Load and split data
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    # Load model
    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Print results
    print("\nModel Evaluation")
    print("-" * 30)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

if __name__ == "__main__":
    main()