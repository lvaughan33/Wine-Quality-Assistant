import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the wine quality dataset."""
    return pd.read_csv(filepath, sep=";")


def split_data(
    df: pd.DataFrame,
    target: str = "quality",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split data into training and testing sets."""

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )