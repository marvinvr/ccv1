import os
from typing import Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None, transform=None, crop_threshold=0.0):
        self.index = x_df.index
        self.filepaths = x_df["filepath"].values
        self.label = y_df.values if y_df is not None else None
        self.classes = y_df.columns.tolist() if y_df is not None else None
        self.transform = transform
        self.crop_threshold = crop_threshold

    def __getitem__(self, index) -> Tuple[str, Image.Image, Union[torch.Tensor, None]]:
        image_id = self.index[index]

        take_cropped = self.crop_threshold > 0.0 and os.path.exists(
            f"../data/processed/cropped/t{self.crop_threshold}/{self.filepaths[index]}"
        )
        if take_cropped:
            image = Image.open(
                f"../data/processed/cropped/t{self.crop_threshold}/{self.filepaths[index]}"
            ).convert("RGB")
        else:
            image = Image.open(f"../data/processed/{self.filepaths[index]}").convert(
                "RGB"
            )

        if self.transform is not None:
            image = self.transform(image)

        label = self.label[index] if self.label is not None else torch.tensor(-1)

        return image_id, image, label

    def __len__(self):
        return len(self.index)
