from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from src.models import interface


class InceptionV3Model(interface.ModelInterface):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        n_classes: int = 8,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        dropout: float = 0.0,
        crop_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            batch_size,
            image_size,
            n_classes,
            lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer="adamw",
            dropout=dropout,
            crop_threshold=crop_threshold,
        )

        self.model = inception_v3(pretrained=True)

        self.model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
        )

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.model.fc.in_features, n_classes)
        )

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            x, _ = output
        else:
            x = output

        return x
