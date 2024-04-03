import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, args, df, df_demo=None):
        self.args = args
        self.df = df
        self.df_demo = df_demo
        self.pcwp_train = np.load("./stores/train_info.npy")
        self.pcwp_mean = self.pcwp_train[0]
        self.pcwp_std = self.pcwp_train[1]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load ECG
        qid = row['QuantaID']
        doc = row['Date_of_Cath']
        fname = os.path.join(self.args.dir_csv, f'{qid}_{doc}.csv')
        x = pd.read_csv(fname).values[::2,1:].astype(np.float32)

        if self.args.train_mode == 'regression':
            y = row['PCWP_mean']
            if self.args.normalize_label:
                y = (y-self.pcwp_mean)/(self.pcwp_std)
        else:
            y = row['PCWP_mean'] > self.args.pcwp_th

        x = x / 1000
        sample = (x[:2496,:].T, y)

        return sample