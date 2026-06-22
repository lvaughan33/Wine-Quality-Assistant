# Wine Quality Assistant

An end-to-end machine learning application that predicts red wine quality scores from physicochemical measurements and provides natural language explanations through a large language model (LLM) interface.

## Project Description

This project uses machine learning to predict wine quality scores based on measurable chemical properties and allows users to interact with the model using natural language. Users can describe a wine conversationally by providing characteristics such as acidity, alcohol content, and sulphate levels. The application extracts the required features, runs the trained model, and returns a predicted quality score with an easy-to-understand explanation.

## Dataset

This project uses the red wine subset of the UCI Wine Quality dataset.

- Source: https://archive.ics.uci.edu/dataset/186/wine+quality
- Records: 1,599 red wine samples
- Features: 11 physicochemical measurements
- Target: Wine quality score (integer values from 3 to 8)

### Features

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

## Project Structure

```text
Wine-Quality-Assistant/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── config.yaml
├── data/
│   └── raw/
├── models/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── select_best_model.py
│   └── train.py
├── tests/
│   ├── test_interface.py
│   ├── test_model.py
│   └── test_preprocess.py
└── pytest.ini
```

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/lvaughan33/Wine-Quality-Assistant.git
cd Wine-Quality-Assistant
```

Create and activate a Conda environment:

```bash
conda create -n wine-ml python=3.11
conda activate wine-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root directory with your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Download `winequality-red.csv` from the UCI Wine Quality dataset website and place it in the following directory:

```text
data/raw/winequality-red.csv
```

## Running the Training Pipeline

Train all machine learning models and log experiments with MLflow:

```bash
python -m src.train
```

Select the best-performing model:

```bash
python -m src.select_best_model
```

Evaluate the selected model:

```bash
python -m src.evaluate
```

Launch the MLflow user interface:

```bash
python -m mlflow ui
```

Then open the following URL in your browser:

```text
http://127.0.0.1:5000
```

## Running the Application

Start the Wine Quality Assistant:

```bash
python -m src.app
```

Example input:

```text
fixed acidity is 7.4, volatile acidity is 0.7, citric acid is 0.0, residual sugar is 1.9, chlorides are 0.076, free sulfur dioxide is 11, total sulfur dioxide is 34, density is 0.9978, pH is 3.51, sulphates are 0.56, alcohol is 9.4
```

Example output:

```text
Predicted wine quality score: 5.3
```

## MLflow Experiment Tracking

MLflow was used to track:

- Model configurations
- Hyperparameters
- Evaluation metrics
- Model artifacts

Five experiments were logged and compared programmatically using `mlflow.search_runs()`.

## Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.504 | 0.625 | 0.403 |
| Ridge Regression | 0.506 | 0.627 | 0.399 |
| Random Forest (100 trees) | 0.422 | 0.549 | 0.539 |
| Random Forest (200 trees) | 0.424 | 0.554 | 0.531 |
| Gradient Boosting | 0.485 | 0.602 | 0.446 |

### Best Model

The Random Forest model with 100 estimators achieved the best performance:

- MAE: 0.422
- RMSE: 0.549
- R²: 0.539

## Testing

Run the complete test suite with:

```bash
pytest tests -v
```

The project includes:

- 4 preprocessing tests
- 2 model tests
- 2 interface tests

## Architecture Overview

1. Raw wine data is loaded and preprocessed.
2. Multiple regression models are trained and tracked with MLflow.
3. The best-performing model is saved locally.
4. The LLM interface extracts feature values from user input.
5. The trained model generates a wine quality prediction.
6. The LLM explains the prediction in natural language.

## Results Summary

The Random Forest model outperformed the linear models and the gradient boosting model, explaining approximately 54% of the variance in wine quality scores.

The LLM interface successfully converts unstructured user input into structured features, enabling non-technical users to interact with the predictive model naturally.

## Reflection

This project demonstrated the complete machine learning lifecycle, including data preprocessing, experiment tracking, model selection, testing, and deployment.

Key challenges included configuring MLflow tracking, integrating an LLM API securely using environment variables, and designing an interface that handles incomplete user input gracefully.

Future improvements could include:

- Deploying the application with Streamlit
- Containerizing the application with Docker
- Adding feature importance visualizations
- Supporting both red and white wine datasets
- Improving prediction accuracy through hyperparameter optimization 