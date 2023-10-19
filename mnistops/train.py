import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .dataset import MNISTDataModule
from .model import MNISTModel


def train(cfg: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = MNISTDataModule(cfg)
    model = MNISTModel(cfg)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/mnistops-logs",
        ),
        pl.loggers.WandbLogger(
            project="mnistops", name=cfg.artifacts.experiment_name
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(
            max_depth=cfg.callbacks.model_summary.max_depth
        ),
    ]

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
