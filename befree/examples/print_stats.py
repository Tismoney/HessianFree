import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

def print_stats(named_stats, step = 50, figsize=(16, 4)):

    stat_names = list(set(chain(*[set(st.keys()) for st in named_stats.values()])))
#     stat_names = ['train.loss', 'train.accuracy', 'test.loss', 'test.accuracy']
    assert len(stat_names) >= 1
    _, ax = plt.subplots(len(stat_names) // 2, 2, figsize=figsize)
    if len(stat_names) // 2 == 1:
        ax = np.array([ax])

    for name, stats in named_stats.items():
        
        for i, st_name in enumerate(stat_names):
            if st_name in stats:
                
                if 'train' in st_name:
                    _stats = np.array(stats[st_name])
                    num_iter = len(_stats)
                    x = np.arange(0, num_iter, step)
                    _stat = [_stats[i:i+step].mean() for i in x]
                    
                    ax[i // 2, i % 2].plot(x, _stat, label=name)
                elif 'test' in st_name:
                    _stat = np.array(stats[st_name])
                    
                    num_iter = len(np.array(stats[st_name.replace('test', 'train')]))
                    x = np.arange(0, num_iter, 50)
                    
                    x, _stat = zip(*(zip(x, _stat)))
                    ax[i // 2, i % 2].plot(x, _stat, label=name)
                
    
    for i, st_name in enumerate(stat_names):
        ax[i // 2, i % 2].set_title(st_name)
        ax[i // 2, i % 2].legend()
        ax[i // 2, i % 2].grid()
        ax[i // 2, i % 2].set_xlabel('Iteration')
        
        
#         if 'loss' in st_name: ax[i // 2, i % 2].set_yscale('log')
#         if 'accuracy' in st_name: 
# #             ax[i // 2, i % 2].set_xscale('log')
#             ax[i // 2, i % 2].set_ylim([0.895, 1.005])

#     plt.show()