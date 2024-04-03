import os
import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve

import torch

from utils.loss import RMSELoss
from utils.recall_at_k import Recall_At_K
from utils.nmi import NMI

metric_names = ['tpr', 'tnr', 'fpr', 'fnr', 'fdr', 'ppv', 'f1', 'auc', 'apr', 'acc', 'loss']
score_to_dict = lambda name, score: dict((name[i], score[i]) for i in range(len(score)))

class Evaluator(object):
    '''Evaluator Object for 
    prediction performance'''

    def __init__(self, args=None):
        if args != None:
            self.args = args
            self.batch_size = args.batch_size
        self.confusion_matrix = np.zeros((2,2))
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.features = []
        self.loss = 0
        self.best_auc = 0
        self.threshold = 0.5
        self.rmse = RMSELoss(args)

        # for statistical testing
        self.loss_list = []
        self.auc_list = []
        self.apr_list = []

        if self.args.metric_learning:
            self.recall_at_1 = Recall_At_K(k=self.args.recallk)
            self.nmi = NMI(args)
        
    def add_batch(self, y_true, y_pred, loss, feature=None, test=False):
        try:
            self.y_true.append(y_true)
            self.y_pred_proba.append(y_pred)
        except:
            self.y_true = np.append(self.y_true, y_true)
            self.y_pred_proba = np.append(self.y_pred_proba, y_pred)

        if self.args.train_mode == 'regression':
            self.y_pred.append(y_pred)
            if test==True:
                self.loss += loss
                self.loss /= 2.0
                
            else:
                self.loss = loss

        elif self.args.train_mode == 'binary_class':
            if self.args.metric_learning:
                self.features.append(feature.detach().cpu().numpy())
            try:
                self.y_pred.append(np.array(y_pred > self.threshold).astype(int))
            except:
                self.y_pred = np.append(self.y_pred, np.array(y_pred > self.threshold).astype(int))
            self.confusion_matrix += confusion_matrix((y_pred > self.threshold), y_true)

    def performance_metric(self, validation=False):
        if self.args.train_mode == 'regression':
            loss = self.loss
            try:
                r, pval = stats.pearsonr(np.concatenate(self.y_true), np.concatenate(self.y_pred))
            except:
                r, pval = stats.pearsonr(np.concatenate(self.y_true), np.concatenate(self.y_pred).squeeze(1))

            if self.args.bootstrap and not validation:
                loss_list, r_list, pval_list = self.do_bootstrap_regression(np.concatenate(self.y_pred), np.concatenate(self.y_true))
                loss_lower, loss_upper = self.confidence_interval(loss_list)
                r_lower, r_upper = self.confidence_interval(r_list)
                pval_lower, pval_upper = self.confidence_interval(pval_list)
                self.loss_list = loss_list

                return np.mean(loss_list), np.mean(r_list), np.mean(pval_list), (loss_lower, loss_upper), (r_lower, r_upper), (pval_lower, pval_upper)
            
            return loss, r, pval
            
        elif self.args.train_mode == 'binary_class':
            try:
                self.y_true = np.concatenate(self.y_true, 0)
                self.y_pred_proba = np.concatenate(self.y_pred_proba, 0)
                self.y_pred = np.concatenate(self.y_pred, 0)
            except:
                pass

            auc = roc_auc_score(self.y_true, self.y_pred_proba)
            apr = average_precision_score(self.y_true, self.y_pred_proba)
            acc = accuracy_score(self.y_true, self.y_pred)
            f1 = f1_score(self.y_true, self.y_pred)

            if self.args.bootstrap and not validation:
                auc_list, apr_list, acc_list, f1_list = self.do_bootstrap(self.y_pred_proba, self.y_pred, self.y_true)
                f1_lower, f1_upper = self.confidence_interval(f1_list)
                auc_lower, auc_upper = self.confidence_interval(auc_list)
                apr_lower, apr_upper = self.confidence_interval(apr_list)
                acc_lower, acc_upper = self.confidence_interval(acc_list)
                self.auc_list = auc_list
                self.apr_list = apr_list

                return np.mean(f1_list), (f1_lower, f1_upper), np.mean(auc_list), (auc_lower, auc_upper), np.mean(apr_list), (apr_lower, apr_upper), np.mean(acc_list), (acc_lower, acc_upper)

            return f1, auc, apr, acc

    def metric_performance(self):
        features = np.concatenate(self.features, axis=0)
        try:
            target_labels = np.array(self.y_true)
        except:
            target_labels = torch.cat(self.y_true, dim=0).numpy()
        return self.recall_at_1(target_labels, features), self.nmi(target_labels, features)
    
    def do_bootstrap(self, preds, pred_vals, trues, n=1000):
        auc_list = []
        apr_list = []
        acc_list = []
        f1_list = []

        rng = np.random.RandomState(seed=1)
        for _ in range(n):
            idxs = rng.choice(len(trues), size=len(trues), replace=True)
            if len(set(trues[idxs])) < 2:
                continue
            pred_arr= preds[idxs]
            true_arr = trues[idxs]
            pred_val_arr = pred_vals[idxs]

            auc = roc_auc_score(true_arr, pred_arr)
            apr = average_precision_score(true_arr, pred_arr)
            acc = accuracy_score(true_arr, pred_val_arr)
            f1 = f1_score(true_arr, pred_val_arr)
            
            auc_list.append(auc)
            apr_list.append(apr)
            acc_list.append(acc)
            f1_list.append(f1)

        return np.array(auc_list), np.array(apr_list), np.array(acc_list), np.array(f1_list)

    def do_bootstrap_regression(self, preds, trues, n=1000):
        rmse_list = []
        r_list = []
        pval_list = []
        
        rng = np.random.RandomState(seed=1)
        for _ in range(n):
            idxs = rng.choice(len(trues), size=len(trues), replace=True)
            pred_arr = preds[idxs]
            true_arr = trues[idxs]
            
            rmse = self.rmse(torch.tensor(pred_arr), torch.tensor(true_arr))
            r, pval = stats.pearsonr(true_arr, pred_arr)
            
            rmse_list.append(rmse)
            r_list.append(r)
            pval_list.append(pval)
        
        return np.array(rmse_list), np.array(r_list), np.array(pval_list)
    
    def confidence_interval(self, values, alpha=0.95):
        lower = np.percentile(values, (1-alpha)/2 * 100)
        upper = np.percentile(values, (alpha + (1-alpha)/2) * 100)
        return lower, upper
    
    def return_pred(self):
        return self.y_pred, self.y_true
    
    def reset(self):
        self.confusion_matrix = np.zeros((2,) * 2)
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.features = []

        if self.args.train_mode == 'regression':
            self.loss = 0.0
        elif self.args.train_mode == "binary_class":
            self.loss = np.inf
            self.threshold = 0.5