import os
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

TRACKING_URI = "file:///C:/Users/lukev/Downloads/Wine-Quality-Assistant/mlruns"

mlflow.set_tracking_uri(TRACKING_URI)

runs = mlflow.search_runs(
    experiment_names=["wine-quality-regression"]
)

best_run = runs.sort_values(
    by="metrics.r2",
    ascending=False
).iloc[0]

print("\nBest Run")
print("-" * 40)
print(f"Run ID: {best_run['run_id']}")
print(f"Model: {best_run['tags.mlflow.runName']}")
print(f"R²: {best_run['metrics.r2']:.3f}")
print(f"RMSE: {best_run['metrics.rmse']:.3f}")
print(f"MAE: {best_run['metrics.mae']:.3f}")