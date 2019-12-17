import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

def print_stats(named_stats, step = 25, figsize=(16, 4)):

    stat_names = list(set(chain(*[set(st.keys()) for st in named_stats.values()])))
    assert len(stat_names) >= 1
    _, ax = plt.subplots(1, len(stat_names), figsize=figsize)
    if len(stat_names) == 1:
        ax = [ax]

    for name, stats in named_stats.items():
        
        for i, st_name in enumerate(stat_names):
            if st_name in stats:
                _stats = np.array(stats[st_name])
                num_iter = len(_stats)
        
                x = np.arange(0, num_iter, step)
                avg_stat = [_stats[i:i+step].mean() for i in x]
                ax[i].plot(x, avg_stat, label=name)
    
    for i, st_name in enumerate(stat_names):
        ax[i].set_title(st_name)
        ax[i].legend()
        ax[i].grid()

    plt.show()