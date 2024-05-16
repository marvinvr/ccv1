from torch import nn
from transformers import ViTForImageClassification
from typing_extensions import override

from src.models import interface


class ViTModel(interface.ModelInterface):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        n_classes: int = 8,
        lr: float = 0.001,
        dropout: float = 0,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        crop_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            batch_size,
            image_size,
            n_classes,
            lr,
            dropout,
            momentum,
            weight_decay,
            optimizer,
            crop_threshold,
        )

        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224-in21k"
        )
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            n_classes,
        )

    @override
    def forward(self, x):
        output = self.model(x)
        return output.logits
