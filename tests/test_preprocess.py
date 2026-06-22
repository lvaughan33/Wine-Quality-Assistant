from pathlib import Path

import pandas as pd

from src.preprocess import load_data, split_data

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "winequality-red.csv"
)

def test_load_data_returns_dataframe():
    df = load_data(DATA_PATH)

    assert isinstance(df, pd.DataFrame)

def test_load_data_has_no_missing_values():
    df = load_data(DATA_PATH)

    assert df.isnull().sum().sum() == 0

def test_split_data_creates_expected_shapes():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    assert X_train.shape[1] == 11
    assert X_test.shape[1] == 11
    assert len(X_train) == 1279
    assert len(X_test) == 320

def test_split_data_does_not_modify_original_dataframe():
    df = load_data(DATA_PATH)

    original_columns = df.columns.tolist()

    split_data(df)

    assert df.columns.tolist() == original_columns