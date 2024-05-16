from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models import interface


class Resnet50Model(interface.ModelInterface):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        output_channels: int,
        n_classes: int = 8,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__(
            batch_size,
            image_size,
            output_channels,
            n_classes,
            lr,
            momentum,
            weight_decay,
        )

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(
                2048, 100
            ),  # dense layer takes a 2048-dim input and outputs 100-dim
            nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
            nn.Dropout(0.1),  # common technique to mitigate overfitting
            nn.Linear(
                100, n_classes
            ),  # final dense layer outputs 8-dim corresponding to our target classes  # softmax activation to convert to probabilities
        )
