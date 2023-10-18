from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from .dataset import get_loaders
from .model import SimpleNet


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    preds = []
    gt = []

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        preds += predictions.cpu().tolist()
        gt += target.cpu().tolist()

    accuracy = accuracy_score(gt, preds)
    return accuracy


def val_epoch(model, device, test_loader):
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
    return accuracy


def train_model(cfg):
    if cfg.artifacts.save_ckpt:
        Path(cfg.artifacts.ckpt_path).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleNet(
        n_classes=cfg.model.n_classes, dropout_rate=cfg.model.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_loaders(
        batch_size=cfg.data.batch_size, root_path=cfg.data.root_path
    )

    best_val_acc = 0

    for _ in range(cfg.train.n_epoch):
        train_accuracy = train_epoch(
            model, device, train_loader, optimizer, criterion
        )

        val_accuracy = val_epoch(model, device, test_loader)

        print(
            f"train_accuracy = {train_accuracy}, \nval_accuracy={val_accuracy}"
        )

        if val_accuracy > best_val_acc:
            savepath = (
                Path(cfg.artifacts.ckpt_path) / cfg.artifacts.best_model_name
            )
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), savepath)
