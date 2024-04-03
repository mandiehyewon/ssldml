import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, args, train_list, **kwargs):
        self.args = args

        #####
        self.train_list = train_list
        self.df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
        self.train_df = self.df_tab[self.df_tab["QuantaID"].isin(self.train_list)].reset_index()
        self.label_list = self.train_df["PCWP_mean"] > self.args.pcwp_th
        
        #####
        self.classes        = list(np.unique(self.label_list))

        ####
        self.batch_size         = self.args.batch_size
        self.samples_per_class  = int(self.batch_size/len(self.classes))#args.samples_per_class
        self.sampler_length     = len(self.train_df)//self.args.batch_size
        assert self.batch_size % self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'class_random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            
            ### Random Subset from Random classes
            for i in range(len(self.classes)):
                class_idxs = np.where(self.label_list == i)[0]
                class_ix_list = [random.choice(class_idxs) for _ in range(self.samples_per_class)]
                subset.extend(class_ix_list)

            yield subset

    def __len__(self):
        return self.sampler_length