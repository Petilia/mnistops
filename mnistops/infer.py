from pathlib import Path

import pytorch_lightning as pl

from .dataset import MNISTDataModule
from .model import MNISTModel


def infer(cfg):
    # Getting best checkpoint name
    best_model_name = (
        Path(cfg.artifacts.checkpoint.dirpath)
        / cfg.artifacts.experiment_name
        / "best.txt"
    )

    with open(best_model_name, "r") as f:
        best_checkpoint_name = f.readline()

    # Getting a torch-model
    model = MNISTModel.load_from_checkpoint(best_checkpoint_name)

    dm = MNISTDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    trainer.test(model, test_loader)
