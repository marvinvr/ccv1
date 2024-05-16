import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassF1Score

from src.models.helpers.metrics import log_loss_metric


class ModelInterface(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        n_classes: int,
        lr: float = 0.001,
        dropout: float = 0.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        optimizer: str = "sgd",
        crop_threshold: float = 0.0,
    ) -> None:
        """
        Initializes the ModelInterface class.

        Args:
            batch_size (int): The number of samples per batch to load.
            image_size (int): The size of the images to load.
            n_classes (int): The number of classes in the dataset.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
            dropout (float, optional): The dropout rate for the model. Defaults to 0.0.
            momentum (float, optional): The momentum for the optimizer. Defaults to 0.9.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.0.
            optimizer (str, optional): The optimizer to use. Defaults to "sgd".
            crop_threshold (float, optional): The threshold for excluding falsely undetected cropping rows. Defaults to 0.

        Returns:
            None
        """
        super().__init__()

        (
            self.lr,
            self.momentum,
            self.batch_size,
            self.image_size,
            self.weight_decay,
            self.optimizer,
            self.dropout,
            self.crop_threshold,
        ) = (
            lr,
            momentum,
            batch_size,
            image_size,
            weight_decay,
            optimizer,
            dropout,
            crop_threshold,
        )
        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss()
        self.f1_weighted = MulticlassF1Score(num_classes=n_classes, average="weighted")
        self.f1_micro = MulticlassF1Score(num_classes=n_classes, average="micro")
        self.f1_macro = MulticlassF1Score(num_classes=n_classes, average="macro")

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _) -> dict[str, torch.Tensor]:
        _, x, y = batch
        self.train(True)
        y_hat = self(x)

        return self._log("train", y_hat, y)

    def validation_step(self, batch, _) -> dict[str, torch.Tensor]:
        _, x, y = batch
        self.train(False)
        y_hat = self(x)

        return self._log("val", y_hat, y)

    def predict_step(self, batch, _) -> dict[str, torch.Tensor]:
        image_id, x, _ = batch
        self.train(False)
        y_hat = self(x)

        return {"image_id": image_id, "prediction": self._softmax(y_hat)}

    def _softmax(self, y_hat):
        """
        Applies the softmax function to the predicted values.

        Args:
            y_hat: The predicted values.

        Returns:
            The predicted values after applying the softmax function.
        """
        return nn.functional.softmax(y_hat, dim=1)

    def _log(self, phase: str, y_hat, y):
        """
        Logs the loss and evaluation metrics for a given phase.

        Args:
            phase (str): The phase to log the metrics for (e.g. "train" or "val").
            y_hat: The predicted values.
            y: The true values.

        Returns:
            The loss value.
        """
        loss = self.loss(y_hat, y)
        preds = self._softmax(y_hat)
        preds_argmax = torch.argmax(preds, dim=1)
        y_argmax = torch.argmax(y, dim=1)

        self.log(f"{phase}_loss", loss)
        self.log(f"{phase}_f1_weighted", self.f1_weighted(preds_argmax, y_argmax))
        self.log(f"{phase}_f1_micro", self.f1_micro(preds_argmax, y_argmax))
        self.log(f"{phase}_f1_macro", self.f1_macro(preds_argmax, y_argmax))
        self.log(f"{phase}_log_loss", log_loss_metric(preds, y))

        return loss

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError("Optimizer not implemented")
