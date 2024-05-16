from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.models import interface


class EfficientnetV2Model(interface.ModelInterface):
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

        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, n_classes
        )
