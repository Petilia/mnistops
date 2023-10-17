import torch
import torchvision

from .constants import DATASET_ROOT


def get_loaders(batch_size=32):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            DATASET_ROOT,
            train=True,
            download=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            DATASET_ROOT,
            train=False,
            download=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
