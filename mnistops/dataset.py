import torch
import torchvision


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
