# HessianFree
Project of Skoltech course "Numerical Linear Algebra"


## Quick quide

```python
from befree.model import get_model
from befree.datasets import get_dataset
from befree.optimizers import get_optimizer
from befree.config import get_config
from befree.examples.train import train

config = get_config('befree/config/mlp.yaml')
train_loader, test_loader = get_dataset(config['dataset'])
model = get_model(config['model'])
optim = get_optimizer(model.parameters(), config['optimizer'])
stats = train(model, train_loader, optim, config['model']['num_epoch'])

from befree.examples.print_stats import print_stats
%matplotlib inline

print_stats({config['optimizer']['name']: stats})
```
