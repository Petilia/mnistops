from pathlib import Path

import torch

from .model import MNISTModel


def export_2_onnx(cfg):
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
