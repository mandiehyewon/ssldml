#!/usr/bin/env python3
import os
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from utils.metrics import Evaluator


class Logger:
    def __init__(self, args, model=None):
        self.args = args
        self.args_save = copy.deepcopy(args)

        # Evaluator
        self.evaluator = Evaluator(self.args)

        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(args.dir_result, args.name)
        self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_save = os.path.join(self.dir_root, 'ckpts')

        self.log_iter = args.log_iter
        
        if not args.finetuning:
            if args.reset and os.path.exists(self.dir_root):
                shutil.rmtree(self.dir_root, ignore_errors=True)
            if not os.path.exists(self.dir_root):
                os.makedirs(self.dir_root)
            if not os.path.exists(self.dir_save):
                os.makedirs(self.dir_save)
            elif os.path.exists(os.path.join(self.dir_save, 'last.pth')) and os.path.exists(self.dir_log):
                shutil.rmtree(self.dir_log, ignore_errors=True)
            if not os.path.exists(self.dir_log):
                os.makedirs(self.dir_log)
        
        if model is not None:
            self.model = model
            self.finetune_ckpt = {'model_bestauc': model, 'model_bestloss': model}

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)
        
        # Log variables
        self.loss = 0

        if self.args.log_metricloss:
            self.metric_loss = 0
            self.triplet_loss = 0

        self.best_auc = 0
        self.bestauc_iter = 0
        self.bestloss_iter = 0
        self.best_results = []
        self.best_loss = 0

        if self.args.metric_learning and self.args.finetuning:
            self.recall_at_k = 0
            self.nmi = 0

    def log_tqdm(self, pbar):
        if self.args.train_mode =='regression':
            tqdm_log = 'loss: {:.5f}, best_loss: {:.5f}, best_iter: {}'.format(self.loss/self.log_iter, self.best_loss, self.bestloss_iter)
        else:
            if self.args.metric_learning and self.args.finetuning:
                tqdm_log = 'loss: {:.5f}, auc: {:.5f}, recall@1: {:.5f}, best_auc: {}, best_loss: {} epc'.format(self.loss/self.log_iter, self.best_auc, self.recall_at_k, self.bestauc_iter, self.bestloss_iter)
                    
            else:
                tqdm_log = 'loss: {:.5f}, auc: {:.5f}, best_auc: {}, best_loss: {} epc'.format(self.loss/self.log_iter, self.best_auc, self.bestauc_iter, self.bestloss_iter)
        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        if self.args.log_metricloss:
            self.writer.add_scalar('metric_loss', self.metric_loss / self.log_iter, global_step=step)
            self.writer.add_scalar('triplet_loss', self.triplet_loss / self.log_iter, global_step=step)
        if self.args.unlabeled and self.args.metric_learning:
            self.writer.add_scalar('pretrained_loss', self.loss / self.log_iter, global_step=step)
        elif self.args.finetuning:
            self.writer.add_scalar('finetuning_loss_{}'.format(self.args.train_mode), self.loss / self.log_iter, global_step=step)
        else:
            self.writer.add_scalar('loss', self.loss / self.log_iter, global_step=step)
            
    def loss_reset(self):
        self.loss = 0

    def add_validation_logs(self, step, loss, feature=None):
        if self.args.train_mode == 'regression':
            loss, r, pval = self.evaluator.performance_metric(validation=True)
            self.writer.add_scalar('val/r', r, global_step=step)
            self.writer.add_scalar('val/pval', pval, global_step=step)
            self.writer.add_scalar('val/rmse_loss', loss, global_step=step)

        else: # classification
            if self.args.metric_learning and self.args.finetuning:
                recall_1, nmi = self.evaluator.metric_performance()
                self.recall_at_k = recall_1
                self.nmi = nmi
                self.writer.add_scalar('val/recall_1', recall_1, global_step=step)
                self.writer.add_scalar('val/nmi', nmi, global_step=step)

            f1, auc, apr, acc = self.evaluator.performance_metric(validation=True)
            if self.best_auc < auc:
                self.bestauc_iter = step
                self.best_auc = auc
                self.best_results = [f1, auc, apr, acc]

            self.writer.add_scalar('val/f1', f1, global_step=step)
            self.writer.add_scalar('val/auroc', auc, global_step=step)
            self.writer.add_scalar('val/auprc', apr, global_step=step)
            self.writer.add_scalar('val/accuracy', acc, global_step=step)
            self.writer.add_scalar('val/ce_loss', loss, global_step=step)

        if self.best_loss == 0.0:
            self.best_loss = loss
            self.bestloss_iter = step
        else:
            if self.best_loss > loss:
                self.best_loss = loss
                self.bestloss_iter = step
            if self.args.train_mode == 'regression':
                self.best_results = [loss, r, pval]
        
        self.writer.flush()

    def save(self, model, optimizer, step, last=None, finetune=None):
        self.model = model
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_results': self.best_results, 'best_step': step, 'last_step' : last}

        if step == self.bestauc_iter:
            if self.args.finetuning:
                self.finetune_ckpt['model_bestauc'] = model
                self.save_ckpt(ckpt, 'bestauc_finetune.pth')
            else:
                self.save_ckpt(ckpt, 'bestauc.pth')

        if step == self.bestloss_iter:
            if self.args.finetuning:
                self.finetune_ckpt['model_bestloss'] = model
                self.save_ckpt(ckpt, 'bestloss_finetune.pth')
            else:
                self.save_ckpt(ckpt, 'bestloss.pth')

        if last:
            if self.args.finetuning:
                self.save_ckpt(ckpt, 'last_finetune.pth')
            else:
                self.save_ckpt(ckpt, 'last.pth')

        if finetune:
            self.save_ckpt(ckpt, 'finetune_{}.pth'.format(self.args.train_mode))
            
        elif step % self.args.save_iter == 0:
            if self.args.finetuning:
                self.save_ckpt(ckpt, 'finetune_{}.pth'.format(step))
            else:
                self.save_ckpt(ckpt, '{}.pth'.format(step))

        return ckpt

    def save_metric(self, model, optimizer, step, last=None):
        '''
        For logging supervised metric learning embedding for finetuning
        For logging embeddings for tsne plotting
        '''
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_results': self.best_results, 'best_step': step, 'last_step' : last}

        self.save_ckpt(ckpt, 'emnbedding_{}.pth'.format(step))

        return ckpt

    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))

    def get_bestmodels(self):
        return self.finetune_ckpt['model_bestauc'], self.finetune_ckpt['model_bestloss']