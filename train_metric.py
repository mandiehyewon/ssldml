import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_data
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from utils.lr_scheduler import LR_Scheduler
import sampler
import criteria
import miner

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
sampler = sampler.select(args.data_sampler, args)
train_loader, val_loader, test_loader = get_data(args, sampler)
model = get_model(args, device=device)
classifier = get_model(args, device=device, valid=True)

batchminer = miner.select(args.batch_miner, args)
to_optim   = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay_rate}]
loss_args  = {'batch':None, 'labels':None}#'idxs':None}
criterion, to_optim = criteria.select(args.loss, args, to_optim, batchminer=batchminer)
val_criterion = get_loss(args)
criterion.to(device)
val_criterion.to(device)

optimizer = optim.Adam(to_optim)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for step, train_batch in enumerate(train_loader):
        train_x, train_y = train_batch
        train_x, train_y = train_x.to(device), train_y.to(device)

        logits = model(train_x)
        loss_args['batch']          = logits
        loss_args['labels']         = train_y
        loss_args['step']           = step

        if args.batch_miner == 'semihard' or args.batch_miner == 'softhard' or args.save_triplet:
            loss_args['train_x'] = train_x
        if args.class_loss:
            train_logits = classifier(logits)
            metric_loss = criterion(**loss_args)
            ce_loss = val_criterion(train_logits.float(), train_y.unsqueeze(1).float())
            loss = args.alpha*metric_loss+ce_loss
            logger.loss += ce_loss.item()
            logger.metric_loss += metric_loss
            logger.triplet_loss += loss

        else:
            loss = criterion(**loss_args)
            logger.triplet_loss += loss        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## LOGGING
    if epoch % args.log_iter == 0:
        logger.log_tqdm(pbar)
        logger.log_scalars(epoch)
        logger.loss_reset()

    ### VALIDATION
    if epoch % args.val_iter == 0:
        model.eval()
        classifier.eval()
        logger.evaluator.reset()
        with torch.no_grad():
            for batch in val_loader:
                val_x, val_y = batch
                val_x, val_y = val_x.to(device), val_y.to(device)

                embed = model(val_x)
                logits = classifier(embed)

                loss = val_criterion(logits.float(), val_y.unsqueeze(1).float())
                logger.evaluator.add_batch(val_y.cpu(), logits.cpu(), loss, embed)
            logger.add_validation_logs(epoch, loss)
        model.train()
        classifier.train()
    logger.save_metric(model, classifier, optimizer, epoch)
    pbar.update(1)

ckpt = logger.save_metric(model, classifier, optimizer, epoch, last=True)
logger.writer.close()

print("\n Finished training.......... Please Start Testing with finetune_downstream.py")