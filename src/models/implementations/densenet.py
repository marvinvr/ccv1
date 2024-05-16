from torch import nn
from torchvision.models import DenseNet161_Weights, densenet161

from src.models import interface


class DensenetModel(interface.ModelInterface):
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

        self.model = densenet161(
            weights=DenseNet161_Weights.DEFAULT,
            drop_rate=dropout,
        )
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            n_classes,
        )
