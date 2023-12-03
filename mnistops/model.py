from pathlib import Path
from typing import Any

import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class SimpleNet(nn.Module):
    def __init__(self, n_classes=10, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(320, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTModel(pl.LightningModule):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SimpleNet(
            n_classes=cfg.model.n_classes, dropout_rate=cfg.model.dropout
        )
        self.loss_fn = nn.CrossEntropyLoss()

        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=cfg.model.n_classes),
                MulticlassF1Score(num_classes=cfg.model.n_classes),
            ]
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, target = batch
        outputs = self(data)
        loss = self.loss_fn(outputs, target)

        self.train_metrics.update(outputs, target)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True
        )

        return {"loss": loss}

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, target = batch
        outputs = self(data)
        loss = self.loss_fn(outputs, target)

        self.val_metrics.update(outputs, target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, target = batch
        outputs = self(data)
        self.test_metrics.update(outputs, target)

    def on_test_epoch_end(self):
        test_metrics = self.test_metrics.compute()
        metric_filename = (
            Path(self.cfg.artifacts.checkpoint.dirpath)
            / self.cfg.loggers.experiment_name
            / "metrics.csv"
        )

        test_metrics = {k: [v.item()] for k, v in test_metrics.items()}

        print(test_metrics)
        pd.DataFrame(test_metrics).to_csv(metric_filename, index=False)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
        )
        return {"optimizer": optimizer}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
