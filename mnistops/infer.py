from pathlib import Path

import pytorch_lightning as pl
from hydra import compose, initialize

from .dataset import MNISTDataModule
from .model import MNISTModel


def run_infer(cfg):
    # Getting best checkpoint name
    best_model_name = (
        Path(cfg.artifacts.checkpoint.dirpath)
        / cfg.loggers.experiment_name
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


def infer(
    config_name: str = "config", config_path: str = "../configs", **kwargs
):
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="mnistops-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )

    run_infer(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py infer`")
