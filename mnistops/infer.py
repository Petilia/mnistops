from pathlib import Path

import torch
from sklearn.metrics import accuracy_score

from .dataset import get_loaders
from .model import SimpleNet


def infer(cfg):
    model = SimpleNet()

    model_path = Path(cfg.artifacts.ckpt_path) / cfg.artifacts.best_model_name
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception(
            "Error: No model checkpoint found. Please train the model first."
        )

    _, test_loader = get_loaders(
        batch_size=cfg.data.batch_size, root_path=cfg.data.root_path
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    preds = []
    gt = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predictions = torch.max(outputs, 1)
            preds += predictions.cpu().tolist()
            gt += target.cpu().tolist()

    accuracy = accuracy_score(gt, preds)

    metric_filename = (
        Path(cfg.artifacts.ckpt_path) / cfg.artifacts.metric_file_name
    )

    with open(metric_filename, "w") as f:
        f.write(f"accuracy={accuracy}")

    print(accuracy)


if __name__ == "__main__":
    infer()
