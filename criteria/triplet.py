import os
import time
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model

# ALLOWED_MINING_OPS  = list(miner.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False
REQUIRES_DEVICE      = True


### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, args, batchminer, device=None):
        super(Criterion, self).__init__()
        self.args = args
        self.margin     = self.args.loss_triplet_margin
        self.batchminer = batchminer
        self.name           = 'triplet'
        self.device = device

        ####
        # self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        
        if self.args.unlabeled:
            self.dist_matrix = np.zeros((args.batch_size, args.batch_size))
            
            if self.args.dtw:
                self.dtw_ckpt = torch.load(os.path.join(self.args.dir_unl, self.args.surr_path), map_location=device)
                self.state = self.dtw_ckpt['model']
                self.dtw_model = get_model(args, device=device, dtw=True)
                self.dtw_model.load_state_dict(self.state)
                self.dtw_model.eval()
            
    def triplet_distance(self, anchor, positive, negative):
        return F.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, step, train_x = None, **kwargs):
        if isinstance(labels, torch.Tensor): 
            labels = labels.cpu().numpy()

        if self.args.batch_miner == 'knn_miner':
            sampled_triplets = self.batchminer(batch, labels, idxs)
        elif self.args.batch_miner == 'dist_miner':
            if self.args.dtw:
                self.calculate_dtw(train_x)
            elif self.args.matrix:
                if self.args.dist_calc == 'euclidean':
                    self.dist_matrix = torch.load(os.path.join(self.args.dir_euc, str(step)+'.pth'))
                elif self.args.dist_calc == 'dtwreal':
                    if self.args.batch_size == 64:
                        self.dist_matrix = torch.load(os.path.join(self.args.dir_dtw_real, str(step)+'.pth'))
                    elif self.args.batch_size == 128:
                        dtw_path = self.args.dir_dtw_real+'_128'
                        self.dist_matrix = torch.load(os.path.join(dtw_path, str(step+1)+'.pth'))
                else:
                    self.dist_matrix = torch.load(os.path.join(self.args.dir_dtw_prev, str(step)+'.pth'))
            else:
                self.calculate_euclidean(train_x)
            sampled_triplets = self.batchminer(batch, self.dist_matrix, step)
        else:
            if self.args.save_triplet:
                sampled_triplets = self.batchminer(batch, labels, train_x=train_x, step=step)
            else:
                sampled_triplets = self.batchminer(batch, labels)

        loss = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)
    
    def calculate_dtw(self, train_x):
        batch_len = len(train_x)
        self.dist_matrix = np.zeros((batch_len, batch_len))
        
        for i in range(batch_len):
            for j in range(i, batch_len):
                if self.dist_matrix[i][j] == 0:
                    dtw_val = self.dtw_model(train_x[i].unsqueeze(0), train_x[j].unsqueeze(0)).cpu().detach().numpy()
                    self.dist_matrix[i][j] = dtw_val
                    self.dist_matrix[j][i] = dtw_val
        return

    def calculate_euclidean(self, train_x):
        batch_len = len(train_x)
        self.dist_matrix = torch.zeros((batch_len, batch_len))
        
        for i in range(batch_len):
            for j in range(i, batch_len):
                if self.dist_matrix[i][j] == 0:
                    dist = torch.mean(torch.sqrt((train_x[i]-train_x[j])**2))
                    self.dist_matrix[i][j] = dist
                    self.dist_matrix[j][i] = dist
        return
