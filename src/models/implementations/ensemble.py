from typing import List

import pytorch_lightning as pl
import torch
from torch import nn


class EnsembleModel(pl.LightningModule):
    def __init__(self, models: List[pl.LightningModule], weights: List[float]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, batch):
        image_id, x, label = batch
        avg_output = torch.zeros(x.size(0), 8, device=self.device)
        for model, weight in zip(self.models, self.weights):
            avg_output += self._softmax(model(x)) * weight
        return image_id, avg_output, label

    def _softmax(self, y_hat):
        return nn.functional.softmax(y_hat, dim=1)
