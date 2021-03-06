# HessianFree

Code of the project "On the New method of Hessian-free second-order optimization".

Made initially as a project for Skoltech course "Numerical Linear Algebra 2019"


## Quick guide

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

## Paper

See our technical report [here](https://drive.google.com/open?id=1qy2nMyWZ8EowgGG99iJHhauXqjlauiIr).
