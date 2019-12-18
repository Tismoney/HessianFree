import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.optimize import rosen
import numpy as np

class RosenData(Dataset):
    def __init__(self, n_dim=3, num_samples=10000,
                 transforms = transforms.ToTensor()):
        self.n_dim = n_dim
        self.num_samples = num_samples
        self.transforms = transforms
        self.init_data()
        
    def init_data(self):
        self.x = 2 * np.random.random((self.num_samples, self.n_dim)) - 1
        self.y = rosen(self.x.T)[:, None]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx, :]
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        
        if self.transforms:
            x = self.transforms(x)
            y = self.transforms(y)
        
        return x, y

def get_rosen(config):
    batch_size = config['batch_size']


    train_dataset = RosenData(config['n_dim'], config['train_size'])
    test_dataset = RosenData(config['n_dim'], config['test_size'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True)

    return train_loader, test_loader