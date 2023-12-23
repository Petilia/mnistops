from pathlib import Path

import torch
from hydra import compose, initialize

from .model import MNISTModel


def run_export_2_onnx(cfg):
    best_model_name = (
        Path(cfg.artifacts.checkpoint.dirpath)
        / cfg.loggers.experiment_name
        / "best.txt"
    )

    with open(best_model_name, "r") as f:
        best_checkpoint_name = f.readline()

    # Getting a torch-model
    model = MNISTModel.load_from_checkpoint(best_checkpoint_name)

    model_name = f"{cfg.export.export_name}.onnx"

    filepath = Path(cfg.export.export_path) / model_name
    filepath.parent.mkdir(parents=True, exist_ok=True)

    input_sample = torch.randn(tuple(cfg.export.input_sample_shape))

    model.to_onnx(
        filepath,
        input_sample,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={
            "IMAGES": {0: "BATCH_SIZE"},
            "CLASS_PROBS": {0: "BATCH_SIZE"},
        },
    )


def export_2_onnx(
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

    run_export_2_onnx(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py export`")
