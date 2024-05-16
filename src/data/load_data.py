import os
import random
from os import cpu_count
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from src.data.custom_dataset import ImagesDataset

from .train_val_split import train_val_split

mean, std = [0.4904, 0.5008, 0.4928], [0.2564, 0.2507, 0.2539]

train_features = pd.read_csv(
    Path("../data/processed/train_features.csv"), index_col="id"
)
train_labels = pd.read_csv(Path("../data/processed/train_labels.csv"), index_col="id")
test_features = pd.read_csv(Path("../data/processed/test_features.csv"), index_col="id")

train_df = train_features.join(train_labels)
train_df.reset_index(inplace=True)
classes = train_labels.columns.tolist()

x_train, y_train, x_val, y_val = train_val_split(train_df)


def get_train_loader(
    batch_size: int, image_size: int, crop_threshold: float = 0
) -> DataLoader:
    """
    Returns a DataLoader object for the training data.

    Args:
        batch_size (int): The number of samples per batch to load.
        image_size (int): The size of the images to load.
        crop_threshold (float, optional): The threshold for excluding falsely undetected cropping rows. Defaults to 0.

    Returns:
        DataLoader: The DataLoader object for the training data.
    """
    return DataLoader(
        ImagesDataset(
            x_train,
            y_train,
            transform=_get_augmented_transform(image_size),
            crop_threshold=crop_threshold,
        ),
        batch_size=batch_size,
        num_workers=cpu_count(),
    )


def get_val_loader(
    batch_size: int, image_size: int, crop_threshold: float = 0
) -> DataLoader:
    """
    Returns a DataLoader object for the validation data.

    Args:
        batch_size (int): The number of samples per batch to load.
        image_size (int): The size of the images to load.
        crop_threshold (float, optional): The threshold for excluding falsely undetected cropping rows. Defaults to 0.

    Returns:
        DataLoader: The DataLoader object for the validation data.
    """
    return DataLoader(
        ImagesDataset(
            x_val,
            y_val,
            transform=_get_transform(image_size),
            crop_threshold=crop_threshold,
        ),
        batch_size=batch_size,
        num_workers=cpu_count(),
    )


def get_test_loader(
    batch_size: int, image_size: int, crop_threshold: float = 0
) -> DataLoader:
    """
    Returns a DataLoader object for the test data.

    Args:
        batch_size (int): The number of samples per batch to load.
        image_size (int): The size of the images to load.
        crop_threshold (float, optional): The threshold for excluding falsely undetected cropping rows. Defaults to 0.

    Returns:
        DataLoader: The DataLoader object for the test data.
    """
    return DataLoader(
        ImagesDataset(
            test_features.filepath.to_frame(),
            transform=_get_transform(image_size),
            crop_threshold=crop_threshold,
        ),
        batch_size=batch_size,
        num_workers=cpu_count(),
    )


def _get_augmented_transform(image_size: int) -> transforms.Compose:
    """
    Returns a Compose object that applies a series of image transformations for data augmentation.

    Args:
        image_size (int): The size of the images to resize to.

    Returns:
        transforms.Compose: The Compose object that applies the image transformations.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # resize the image to 224x224
            transforms.RandomHorizontalFlip(),  # flip the image horiozontally with 50% probability
            transforms.ColorJitter(
                brightness=0, contrast=0, saturation=0, hue=0
            ),  # randomly change the brightness, contrast and saturation of an image
            transforms.Grayscale(3),  # convert to grayscale with 3 channels
            transforms.ToTensor(),  # convert the image to a tensor
            transforms.Normalize(mean, std),  # normalize the image
        ]
    )


def _get_transform(image_size: int) -> transforms.Compose:
    """
    Returns a Compose object that applies a series of image transformations.

    Args:
        image_size (int): The size of the images to resize to.

    Returns:
        transforms.Compose: The Compose object that applies the image transformations.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),  # resize the image to the specified size
            transforms.Grayscale(3),  # convert to grayscale with 3 channels
            transforms.ToTensor(),  # convert the image to a tensor
            transforms.Normalize(mean, std),  # normalize the image
        ]
    )
