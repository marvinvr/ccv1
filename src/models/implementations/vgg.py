from torch import nn
from torchvision.models import VGG19_BN_Weights, vgg19_bn

from src.models import interface


class VGG19Model(interface.ModelInterface):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        output_channels: int,
        n_classes: int = 8,
        lr: float = 0.001,
        momentum: float = 0.9,
        n: int = 0,
    ) -> None:
        super().__init__(
            batch_size,
            image_size,
            output_channels=output_channels,
            n_classes=n_classes,
            lr=lr,
            momentum=momentum,
        )

        self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        for i in range(-n, 0):
            for param in self.model.features[i].parameters():
                param.requires_grad = True

        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, n_classes
        )
