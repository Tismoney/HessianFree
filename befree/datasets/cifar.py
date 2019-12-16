import torch
import torchvision
from torchvision import datasets, transforms


def get_cifar(config):
    batch_size = config['batch_size']

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10('./loaded', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./loaded', train=False,
                                     download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True)

    return train_loader, test_loader