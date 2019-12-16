import numpy as np
from matplotlib import pyplot as plt


def print_stats(named_stats, step = 25, figsize=(16, 4)):
    _, ax = plt.subplots(1, 2, figsize=figsize)
    
    for name, stats in named_stats.items():
        stats = np.array(stats)
        num_iter = stats.shape[0]
        
        x = np.arange(0, num_iter, step)
        avg_loss = [stats.T[0][i:i+step].mean() for i in x]
        avg_acc = [stats.T[1][i:i+step].mean() for i in x]

        
        ax[0].plot(x, avg_loss, label=name)
        ax[1].plot(x, avg_acc, label=name)
        
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    plt.show()