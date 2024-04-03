import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import f_oneway

import torch
import torch.nn as nn

def set_seeds(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def set_devices(args):
    if args.cpu or not torch.cuda.is_available():
        print('using cpu')
        return torch.device('cpu')
    else:
        return torch.device('cuda')

def logit2prob(logit):
    return np.ones(logit.shape)/(np.ones(logit.shape) + np.exp(logit))

def scatterplot(args, x,y):
    plt.scatter(x, y)
    plt.savefig(os.path.join(args.dir_result, args.name, args.name+'.png'))

    return

def stat_testing(list1, list2, list3=None, list4=None):
    if list3 is not None:
        kruskal_stat, kruskal_p_value = stats.kruskal(list1, list2, list3, list4)
        anova_stat, anova_p_value = f_oneway(list1, list2, list3, list4)
        return kruskal_stat, kruskal_p_value, anova_stat, anova_p_value
    
    else: # sex performance gap
        kruskal_stat, kruskal_p_value = stats.kruskal(list1, list2)
        t_stat, t_p_value = stats.ttest_ind(list1, list2)
        return kruskal_stat, kruskal_p_value, t_stat, t_p_value