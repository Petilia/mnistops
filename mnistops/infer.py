from pathlib import Path

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Getting a torch-model
    model = MNISTModel.load_from_checkpoint(best_checkpoint_name).model
    model.to(device)
    model.eval()

    # Getting a test dataloader
    dm = MNISTDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()

    metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=cfg.model.n_classes),
            MulticlassF1Score(num_classes=cfg.model.n_classes),
        ]
    )

    # Simulating an inference
    with torch.inference_mode():
        for batch in test_loader:
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            metrics.update(outputs.detach().cpu(), target.detach().cpu())

    metric = metrics.compute()
    print(metric)

    # Saving metric to file
    acc = metric["MulticlassAccuracy"]
    f1 = metric["MulticlassF1Score"]

    metric_filename = (
        Path(cfg.artifacts.checkpoint.dirpath)
        / cfg.artifacts.experiment_name
        / "metrics.txt"
    )

    with open(metric_filename, "w") as f:
        f.write(f"accuracy={acc}, f1={f1}")


# if __name__ == "__main__":
#     infer()
