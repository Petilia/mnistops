from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from .constants import (
    BATCH_SIZE,
    BEST_MODEL_NAME,
    CKPT_PATH,
    N_EPOCH,
    SAVE_CKPT,
)
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


def val_epoch(model, device, test_loader, optimizer, criterion):
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


def train_model():
    if SAVE_CKPT:
        Path(CKPT_PATH).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    best_val_acc = 0

    for _ in range(N_EPOCH):
        train_accuracy = train_epoch(
            model, device, train_loader, optimizer, criterion
        )
        val_accuracy = val_epoch(
            model, device, test_loader, optimizer, criterion
        )
        print(
            f"train_accuracy = {train_accuracy}, \nval_accuracy={val_accuracy}"
        )

        if val_accuracy > best_val_acc:
            savepath = Path(CKPT_PATH) / BEST_MODEL_NAME
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), savepath)


if __name__ == "__main__":
    train_model()
