import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_unlabeleddata
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from utils.lr_scheduler import LR_Scheduler
import sampler
import criteria
import miner
 
'''
CUDA_VISIBLE_DEVICES=2 python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-miner dist_miner --dtw --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 100 --name unl10k_batch64_embed128_100epc

CUDA_VISIBLE_DEVICES=2 python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-miner dist_miner --dtw --matrix --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 100 --name unl_matrix_batch64_embed128_100epc

DTW Real
Save Triplet:
CUDA_VISIBLE_DEVICES=2 python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-miner dist_miner --matrix --dist-calc dtwreal --save-triplet --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 100 --name unl_dtwrealmatrix_batch64_embed128_100epc

CUDA_VISIBLE_DEVICES=2 python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-miner dist_miner --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 100 --name unl10k_euc_batch64_embed128_100epc

CUDA_VISIBLE_DEVICES=2 python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-miner dist_miner --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 100 --name unl_euc_batch64_embed128_100epc
Train with unlabeled data, then test with labeled dataset
'''

# Set seed, device, logger
seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
train_loader = get_unlabeleddata(args)
model = get_model(args, device=device)

if args.load_model:
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/{}.pth'.format(args.load_epoch))
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model']
    model.load_state_dict(state)

batchminer = miner.select(args.batch_miner, args)
to_optim   = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay_rate}]
loss_args  = {'batch':None, 'labels':None}#'idxs':None}
criterion, to_optim = criteria.select(args.loss, args, to_optim, batchminer=batchminer, device= device)
criterion.to(device)

optimizer = optim.Adam(to_optim)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, len(train_loader), from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    
    for step, train_batch in enumerate(train_loader):
        if args.dtw:
            train_x, idx = train_batch
        else:
            train_x = train_batch
        train_x = train_x.to(device)
        logits = model(train_x)

        loss_args['train_x'] = train_x
        loss_args['batch'] = logits
        loss_args['step'] = step
        loss = criterion(**loss_args)
        logger.loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## LOGGING
    logger.log_tqdm(pbar)
    logger.log_scalars(epoch)
    logger.loss_reset()
    logger.save(model, optimizer, epoch)
    
    pbar.update(1)
    
ckpt = logger.save(model, optimizer, epoch, last=True)
logger.writer.close()

print("\n Finished pretraining.......... Please Start Finetuning and Testing with finetune_downstream.py")