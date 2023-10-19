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

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = torchvision.datasets.MNIST(
                self.root_path,
                train=True,
                download=False,
                transform=self.transform,
            )

            self.val_dataset = torchvision.datasets.MNIST(
                self.root_path,
                train=False,
                download=False,
                transform=self.transform,
            )

        if stage == "predict":
            self.test_dataset = torchvision.datasets.MNIST(
                self.root_path,
                train=False,
                download=False,
                transform=self.transform,
            )

        print(f"stage = {stage}")

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

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5,
        )
