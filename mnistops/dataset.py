from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.root_path = cfg.data.root_path
        self.batch_size = cfg.data.batch_size
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = torchvision.datasets.MNIST(
            self.root_path,
            train=True,
            download=False,
            transform=self.transform,
        )

        print(len(self.train_dataset))

        self.val_dataset = torchvision.datasets.MNIST(
            self.root_path,
            train=False,
            download=False,
            transform=self.transform,
        )

        print(len(self.val_dataset))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=5,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5,
        )


def get_loaders(batch_size=32, root_path="./data/"):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root_path,
            train=True,
            download=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root_path,
            train=False,
            download=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
