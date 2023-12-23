from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from .dataset import MNISTDataModule
from .model import MNISTModel


def run_training(cfg: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = MNISTDataModule(cfg)
    model = MNISTModel(cfg)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.loggers.experiment_name,
            tracking_uri=cfg.loggers.mlflow.tracking_uri,
            artifact_location=cfg.loggers.mlflow.artifact_location,
            save_dir=cfg.loggers.mlflow.save_dir,
            log_model=cfg.loggers.mlflow.log_model,
            tags=cfg.loggers.mlflow.tags,
        ),
        pl.loggers.WandbLogger(
            project=cfg.loggers.wandb.project,
            name=cfg.loggers.experiment_name,
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(
            max_depth=cfg.callbacks.model_summary.max_depth
        ),
    ]

    if cfg.artifacts.checkpoint.use:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.artifacts.checkpoint.dirpath)
            / cfg.loggers.experiment_name,
            filename=cfg.artifacts.checkpoint.filename,
            monitor=cfg.artifacts.checkpoint.monitor,
            mode=cfg.artifacts.checkpoint.mode,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            verbose=True,
        )

        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_epochs=cfg.train.n_epoch,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=dm)

    # save best model path to .txt file
    # (it is necessary to use best model in inference)
    if cfg.artifacts.checkpoint.use:
        print(checkpoint_callback.best_model_path)

        best_model_name = (
            Path(cfg.artifacts.checkpoint.dirpath)
            / cfg.loggers.experiment_name
            / "best.txt"
        )

        with open(best_model_name, "w") as f:
            f.write(checkpoint_callback.best_model_path)


def train(
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

    run_training(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py train`")
