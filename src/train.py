import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_data, split_data

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2

def build_model(config):
    """Create a model from the configuration."""

    name = config["name"]

    if name == "linear_regression":
        return LinearRegression()

    if name == "ridge":
        return Ridge(alpha=config["alpha"])

    if name in ["random_forest_100", "random_forest_300"]:
        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
        )

    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")

def main():

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    tracking_uri = (
        "file:///C:/Users/lukev/Downloads/Wine-Quality-Assistant/mlruns"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wine-quality-regression")

    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    df = load_data(config["data"]["raw_path"])

    X_train, X_test, y_train, y_test = split_data(
        df,
        target=config["model"]["target"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    best_r2 = float("-inf")
    best_model = None

    for model_config in config["experiments"]["models"]:

        model = build_model(model_config)

        with mlflow.start_run(run_name=model_config["name"]):

            model.fit(X_train, y_train)

            mae, rmse, r2 = evaluate_model(
                model,
                X_test,
                y_test,
            )

            mlflow.log_params(model_config)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
            )

            print(
                f"{model_config['name']}: "
                f"MAE={mae:.3f}, "
                f"RMSE={rmse:.3f}, "
                f"R²={r2:.3f}"
            )

            if r2 > best_r2:
                best_r2 = r2
                best_model = model

    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\nBest R²: {best_r2:.3f}")

if __name__ == "__main__":
    main()