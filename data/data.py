import os, sys
import itertools
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from config import args
from .apollo import ECGDataset
from .unl_ecg import UNLECGDataset

def get_data(args, sampler=None):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
    train_ids = np.load("./stores/train_ids.npy")
    val_ids = np.load("./stores/val_ids.npy")
    test_ids = np.load("./stores/test_ids.npy")

    if args.pretraining:
        '''
        Pretraining for supervised deep metric learning.
        Use the first half indices of train/val/test_ids
        '''
        train_ids = train_ids[:len(train_ids)//2]
        val_ids = val_ids[:len(val_ids)//2]
        test_ids = test_ids[:len(test_ids)//2]

    elif args.finetuning and args.metric_learning:
        '''
        Finetuning for supervised deep metric learning.
        Use the second half indices of train/val/test_ids
        '''
        train_ids = train_ids[len(train_ids)//2:]
        val_ids = val_ids[len(val_ids)//2:]
        test_ids = test_ids[len(test_ids)//2:]

    if args.limited: # only using 1,000 data point
        train_ids = train_ids[:600]
        val_ids = val_ids[:200]
        test_ids = test_ids[:200]

    train_df = df_tab[df_tab["QuantaID"].isin(train_ids)]
    val_df = df_tab[df_tab["QuantaID"].isin(val_ids)]
    test_df = df_tab[df_tab["QuantaID"].isin(test_ids)]

    train_loader = DataLoader(
        ECGDataset(args, train_df),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=16,
        drop_last=True
    )        
    val_loader = DataLoader(
        ECGDataset(args, val_df),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=16,
        drop_last=True
    )
    test_loader = DataLoader(
        ECGDataset(args, test_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        # drop_last=True
    )
    
    return train_loader, val_loader, test_loader

def get_testdata(args):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
    train_ids = np.load("./stores/train_ids.npy")
    test_ids = np.load("./stores/test_ids.npy")
    train_df = df_tab[df_tab["QuantaID"].isin(train_ids)]
    test_df = df_tab[df_tab["QuantaID"].isin(test_ids)]

    print(len(train_df), len(test_df))
    train_label = np.asarray(train_df['PCWP_mean'])
    test_label = np.asarray(test_df['PCWP_mean'])
    fname = os.path.join(args.dir_csv, '{}_{}.csv'.format(df_tab.iloc[0]['QuantaID'], df_tab.iloc[0]['Date_of_Cath']))
    test_x = pd.read_csv(fname).values[::2,1:].astype(np.float32)
    test_x = np.expand_dims(test_x, axis=0)

    for idx in range(1,len(test_label)):
        row = df_tab.iloc[idx]
        # load ECG
        qid = row['QuantaID']
        doc = row['Date_of_Cath']

        fname = os.path.join(args.dir_csv, f'{qid}_{doc}.csv')
        x = pd.read_csv(fname).values[::2,1:].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        test_x = np.concatenate((test_x, x), axis=0)

    print(test_x.shape)
    return train_label, test_x, test_label

def get_unlabeleddata(args): 
    '''
    output unlabeled data for unlabeled deep metric learning
    Get ecg IDs from the numpy 
    '''
    if args.matrix:
        ecg_ids = np.load(os.path.join(args.dir_unl, "ecgno_batch.npy"))
    else:
        ecg_ids = np.load(os.path.join(args.dir_unl, "ecgno_final.npy"))[:args.unl_size] # 06/18 including only 1k dataset

    train_loader = DataLoader(
        UNLECGDataset(args, ecg_ids),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=80,
        drop_last=args.drop_last
    )

    return train_loader