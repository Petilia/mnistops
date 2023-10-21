from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision
from sklearn.model_selection import train_test_split


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

            # There are only 2 parts of the dataset in MNIST - train and val.
            # Therefore, 2 samples are generated from val - val and test.
            full_val_dataset = torchvision.datasets.MNIST(
                self.root_path,
                train=False,
                download=False,
                transform=self.transform,
            )

            val_idx, _ = train_test_split(
                range(len(full_val_dataset)), test_size=0.3, random_state=42
            )

            self.val_dataset = torch.utils.data.Subset(
                full_val_dataset, val_idx
            )

            print(f"train_dataset_size = {len(self.train_dataset)}")
            print(f"val_dataset_size = {len(self.val_dataset)}")

        if stage == "predict":
            full_val_dataset = torchvision.datasets.MNIST(
                self.root_path,
                train=False,
                download=False,
                transform=self.transform,
            )

            _, test_idx = train_test_split(
                range(len(full_val_dataset)), test_size=0.3, random_state=42
            )

            self.test_dataset = torch.utils.data.Subset(
                full_val_dataset, test_idx
            )

            print(f"test_dataset_size = {len(self.test_dataset)}")

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
