"""Class for splitting the dataset into train/val and test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split


def basic_split(df: pd.DataFrame, train_size: float):
    """
    :param df: dataframe containing data to be split.
    :param train_size: train size for split.
    :return: dataframe containing train, val, and test data.
    """
    return train_test_split(df, train_size=train_size, random_state=42)
