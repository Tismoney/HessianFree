import torch
import torchvision
from torchvision import datasets, transforms


def get_mnist(config):
    batch_size = config['batch_size']

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = datasets.MNIST('./loaded', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.MNIST('./loaded', train=False,
                                     download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True)

    return train_loader, test_loader