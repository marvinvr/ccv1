import random
from typing import Tuple

import pandas as pd


def train_val_split(
    train_df, train_size: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the training data into training and validation sets.

    Args:
        train_df (pd.DataFrame): The training data to split.
        train_size (float, optional): The proportion of the data to use for training. Defaults to 0.8.
        random_state (int, optional): The random seed to use for shuffling the data. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation data.
    """
    site_ids = train_df["site"].unique()

    # site_ids in train and val set
    random.seed(random_state)
    random.Random(100).shuffle(site_ids)

    train_site_ids = site_ids[: int(train_size * len(site_ids))]  # ~80% train
    val_site_ids = site_ids[int(train_size * len(site_ids)) :]  # ~20% test

    train_data = train_df[train_df["site"].isin(train_site_ids)]
    val_data = train_df[train_df["site"].isin(val_site_ids)]

    train_data.set_index("id", inplace=True)
    val_data.set_index("id", inplace=True)

    x_train = train_data.iloc[:, :1]
    y_train = train_data.iloc[:, 2:]
    x_val = val_data.iloc[:, :2]
    y_val = val_data.iloc[:, 2:]

    return x_train, y_train, x_val, y_val
