import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim

from config import args
from data import get_unlabeleddata
from model import get_model
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from utils.loss import obtain_contrastive_loss

'''
Pretraining contrastive learning baseline for SimLCR and CLOCS
Configs: refer to config.py CLOCS Paramters section
CUDA_VISIBLE_DEVICES=3 python pretrain_clocs.py --unlabeled --model cnn --label pcwp --contrastive --embedding-dim 128 --contrastive-mode simclr --name simclr_128 --gaussian
CUDA_VISIBLE_DEVICES=1 python pretrain_clocs.py --unlabeled --model cnn --label pcwp --contrastive --embedding-dim 128 --contrastive-mode cmsc --name cmsc_128 --gaussian
CUDA_VISIBLE_DEVICES=1 python pretrain_clocs.py --unlabeled --model cnn --label pcwp --contrastive --embedding-dim 128 --contrastive-mode tfc --name tfc_128
'''

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
train_loader = get_unlabeleddata(args)
model = get_model(args, device=device)

to_optim   = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay_rate}]
optimizer = optim.Adam(to_optim)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for step, train_batch in enumerate(train_loader):
        train_x, pids = train_batch
        train_x = train_x.to(device).type(torch.cuda.FloatTensor)

        latent = []
        for n in range(args.nviews):
            outputs = model(train_x[:,:,:,n])
            latent.append(outputs)
        latent_embed = torch.stack(latent, dim=-1)

        loss = obtain_contrastive_loss(args, latent_embed, pids)
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


print("\n Finished training.......... Please Start Testing with test.py")