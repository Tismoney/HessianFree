# HessianFree
Project of Skoltech course "Numerical Linear Algebra"




## Quick quide

```python
from befree.examples.utils import fit_model
from befree.examples.print_stats import print_stats
%matplotlib inline

configs = ['befree/config/saved_configs/mnist_mlp_adam.yaml',
           'befree/config/saved_configs/mnist_mlp_sgd.yaml',
           'befree/config/saved_configs/mnist_mlp_lbfgs.yaml']

stats = dict([fit_model(config) for config in configs])

print_stats(stats)
```
