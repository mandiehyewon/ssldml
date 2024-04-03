import os
import copy
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="The input object of type 'Tensor' is an array-like implementing one of the corresponding protocols")

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_data
from model import get_model, TFC, target_classifier
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices, stat_testing, trend_pred
from utils.lr_scheduler import LR_Scheduler

'''
Example Run: 
Supervised DML, embedding dimension 128, finetuning for regression
CUDA_VISIBLE_DEVICES=1 python finetune_downstream.py --name sup_pretrain_metric_epc300_embed128 --model cnn --finetuning --train-mode regression --last --embedding-dim 128 --normalize-label --bootstrap --reset --sex-perf

Unsupervised DML, embedding dimension 128, finetuning for regression
CUDA_VISIBLE_DEVICES=3 python finetune_downstream.py --name unl_dtwrealmatrix_batch64_negrandom_embed128_100epc --model cnn --finetuning --train-mode regression --last --embedding-dim 128 --unlabeled --normalize-label --bootstrap --sex-perf --reset --metric-learning

SimCLR, CLOCS baseline
CUDA_VISIBLE_DEVICES=3 python finetune_downstream.py --name clocs --model cnn --finetuning --train-mode regression --last --embedding-dim 128 --unlabeled --normalize-label --bootstrap --sex-perf --reset --metric-learning
'''

seed = set_seeds(args)
device = set_devices(args)

# Load Data, Create Model
if args.sex_perf and args.age_perf:
    train_loader, val_loader, test_loader, male_loader, female_loader, age1_loader, age2_loader, age3_loader, age4_loader = get_data(args)
elif args.sex_perf:
    train_loader, val_loader, test_loader, male_loader, female_loader = get_data(args)
elif args.age_perf:
    train_loader, val_loader, test_loader, age1_loader, age2_loader, age3_loader, age4_loader = get_data(args)
else:
    train_loader, val_loader, test_loader = get_data(args)

## Load DML Model Checkpoint
if args.reset:
    print('testing again:')
if args.last:
    print('testing from the last epoch')
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/last.pth')
    trend_name = 'trend_last'
elif args.best_auc:
    print('testing from the epoch with the best AUC')
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/bestauc.pth')
    trend_name = 'trend_best_auc'
elif args.best_loss:
    print('testing from the epoch with the best loss') 
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/bestloss.pth')
    trend_name = 'trend_best_loss'
elif args.load_epoch is not None:
    print('testing from the epoch with epoch {}'.format(args.load_epoch))
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/{}.pth'.format(args.load_epoch))
    trend_name = 'trend_epoch_{}'.format(args.load_epoch)

# Check if checkpoint exists
if not os.path.exists(ckpt_path):
    print("invalid checkpoint path : {}".format(ckpt_path))

if args.contrastive_mode == 'tfc':
    model = TFC(args).to(device)
    classifier = target_classifier(args).to(device)
else:
    dml_model = get_model(args, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model']
    dml_model.load_state_dict(state)
    dml_model.train()

    model = get_model(args, device=device, finetuning=args.finetuning, dml_model=dml_model)

logger = Logger(args, model=model)
criterion = get_loss(args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

if args.normalize_label:
    pcwp_train = np.load("./stores/train_info.npy")
    pcwp_mean = pcwp_train[0]
    pcwp_std = pcwp_train[1]

print('loaded model '+args.name)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for train_batch in train_loader:
        if args.contrastive_mode == 'tfc':
            data, train_y, aug1, data_f, aug1_f = train_batch
            data, train_y, aug1, data_f, aug1_f = data.float().to(device), train_y.long().to(device), aug1.float().to(device), data_f.float().to(device), aug1_f.float().to(device)
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            logits = classifier(fea_concat)

        else:
            train_x, train_y = train_batch
            train_x, train_y = train_x.to(device), train_y.to(device)
            logits = model(train_x)
        
        if args.normalize_label: # for normalized label for regression task
            logits = logits*pcwp_std+pcwp_mean
            train_y = train_y*pcwp_std+pcwp_mean
        
        loss = criterion(logits.float(), train_y.unsqueeze(1).float())
        logger.loss += loss.item()

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
        logger.evaluator.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                if args.contrastive_mode == 'tfc':
                    val_x, val_y, aug1, data_f, aug1_f = batch
                    val_x, val_y, aug1, data_f, aug1_f = val_x.float().to(device), val_y.long().to(device), aug1.float().to(device), data_f.float().to(device), aug1_f.float().to(device)
                    h_t, z_t, h_f, z_f = model(val_x, data_f)
                    feature = torch.cat((z_t, z_f), dim=1)
                    logits = classifier(feature)

                else:
                    val_x, val_y = batch
                    val_x, val_y = val_x.to(device), val_y.to(device)

                    logits = model(val_x)
                    feature = model(val_x, get_embedding=True)
                
                if args.normalize_label: # for normalized label for regression task
                    logits = logits*pcwp_std+pcwp_mean
                    val_y = val_y*pcwp_std+pcwp_mean
                    
                loss = criterion(logits.float(), val_y.unsqueeze(1).float())
                
                if args.metric_learning:
                    logger.evaluator.add_batch(val_y.cpu(), logits.cpu(), loss, feature=feature)
                else:
                    logger.evaluator.add_batch(val_y.cpu(), logits.cpu(), loss)
                logger.add_validation_logs(epoch, loss)
                logger.save(model, optimizer, epoch)
        model.train()
    pbar.update(1)

# Saving the finetuned model
ckpt = logger.save(model, optimizer, epoch, finetune=True)
logger.writer.close()

print("\n Finished training.......... Start Testing")

# Check if result exists
if args.load_epoch is not None:
    result_ckpt = os.path.join(args.dir_result, args.name, 'test_result_{}_{}.pth'.format(args.train_mode, args.load_epoch))
    result_bestauc_ckpt = os.path.join(args.dir_result, args.name, 'test_result_bestauc_{}_{}.pth'.format(args.train_mode, args.load_epoch))
    result_bestloss_ckpt = os.path.join(args.dir_result, args.name, 'test_result_bestloss_{}_{}.pth'.format(args.train_mode, args.load_epoch))
else:
    result_ckpt = os.path.join(args.dir_result, args.name, 'test_result_{}.pth'.format(args.train_mode))
    
if (not args.reset) and os.path.exists(result_ckpt):
    print('this experiment has tested before.')
    sys.exit()

# Load the model with the best epoch
model_bestauc, model_bestloss = logger.get_bestmodels()

if args.train_mode == 'binary_class':
    model_bestauc.eval()

# Load the model with the best loss
model_bestloss.eval()

# Load the evaluators for both best epoch and best loss
if args.train_mode == 'binary_class':
    evaluator_bestauc = Evaluator(args)
evaluator_bestloss = Evaluator(args)

model.eval()
logger.evaluator.reset()

def downstream_test(test_loader, args, device, model, model_bestauc, model_bestloss, logger):
    rmse_list = []
    auc_list = []
    apr_list = []
    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            if args.contrastive_mode == 'tfc':
                test_x, test_y, aug1, data_f, aug1_f = train_batch
                test_x, test_y, aug1, data_f, aug1_f = test_x.float().to(device), test_y.long().to(device), aug1.float().to(device), data_f.float().to(device), aug1_f.float().to(device)
                h_t, z_t, h_f, z_f = model(test_x, data_f)
                feature = torch.cat((z_t, z_f), dim=1)
                logits = classifier(feature)
                
            else:
                test_x, test_y = test_batch
                test_x, test_y = test_x.to(device), test_y.to(device)
                
                logits = model(test_x)
                feature = model(test_x, get_embedding=True)

            if args.train_mode == 'binary_class':
                if args.contrastive_mode == 'tfc':
                    h_t, z_t, h_f, z_f = model_bestauc(test_x, data_f)
                    logits_bestauc = classifier(torch.cat((z_t, z_f), dim=1))
                else:
                    logits_bestauc = model_bestauc(test_x)
            if args.contrastive_mode == 'tfc':
                    h_t, z_t, h_f, z_f = model_bestloss(test_x, data_f)
                    logits_bestloss = classifier(torch.cat((z_t, z_f), dim=1))
            else:
                logits_bestloss = model_bestloss(test_x)
                
            if args.normalize_label: # for normalized label for regression task
                logits = logits*pcwp_std+pcwp_mean
                if args.train_mode == 'binary_class':
                    logits_bestauc = logits_bestauc*pcwp_std+pcwp_mean
                logits_bestloss = logits_bestloss*pcwp_std+pcwp_mean
                test_y = test_y*pcwp_std+pcwp_mean

            loss = criterion(logits.float(), test_y.unsqueeze(1).float())
            if args.train_mode == 'binary_class':
                loss_bestauc = criterion(logits_bestauc.float(), test_y.unsqueeze(1).float())
            loss_bestloss = criterion(logits_bestloss.float(), test_y.unsqueeze(1).float())

            logger.evaluator.add_batch(test_y.cpu(), logits.cpu(), loss, feature=feature, test=True)
            if args.train_mode == 'binary_class':
                evaluator_bestauc.add_batch(test_y.cpu(), logits_bestauc.cpu(), loss_bestauc, feature=feature, test=True)
            evaluator_bestloss.add_batch(test_y.cpu(), logits_bestloss.cpu(), loss_bestloss, feature=feature,test=True)
            
        if args.train_mode == 'binary_class':
            f1, (f1_lower, f1_upper), auc, (auc_lower, auc_upper), apr, (apr_lower, apr_upper), acc, (acc_lower, acc_upper) = logger.evaluator.performance_metric()
            f1_bestauc, (f1_bestauclower, f1_bestaucupper), auc_bestauc, (auc_bestauclower, auc_bestaucupper), apr_bestauc, (apr_bestauclower, apr_bestaucupper), acc_bestauc, (acc_bestauclower, acc_bestaucupper) = evaluator_bestauc.performance_metric()
            f1_bestloss, (f1_bestlosslower, f1_bestlossupper), auc_bestloss, (auc_bestlosslower, auc_bestlossupper), apr_bestloss, (apr_bestlosslower, apr_bestlossupper), acc_bestloss, (acc_bestlosslower, acc_bestlossupper) = evaluator_bestloss.performance_metric()
            
            print ('f1: {} ({}, {}), auc: {} ({}, {}), apr: {} ({}, {}), acc: {} ({}, {})'.format(f1, f1_lower, f1_upper, auc, auc_lower, auc_upper, apr, apr_lower, apr_upper, acc, acc_lower, acc_upper))
            print('==========================Best AUC==========================')
            print('best auc: {} epoch'.format(logger.bestauc_iter))
            print ('f1: {} ({}, {}), auc: {} ({}, {}), apr: {} ({}, {}), acc: {} ({}, {})'.format(f1_bestauc, f1_bestauclower, f1_bestaucupper, auc_bestauc, auc_bestauclower, auc_bestaucupper, 
                                                                                                    apr_bestauc, apr_bestauclower, apr_bestaucupper, acc_bestauc, acc_bestauclower, acc_bestaucupper))
            print('==========================Best Loss==========================')
            print('best loss: {} epoch'.format(logger.bestloss_iter))
            print ('f1: {} ({}, {}), auc: {} ({}, {}), apr: {} ({}, {}), acc: {} ({}, {})'.format(f1_bestloss, f1_bestlosslower, f1_bestlossupper, auc_bestloss, auc_bestlosslower, auc_bestlossupper, 
                                                                                                    apr_bestloss, apr_bestlosslower, apr_bestlossupper, acc_bestloss, acc_bestlosslower, acc_bestlossupper))
            result_dict = {'auc': auc, 'apr': apr, 'acc': acc, 'f1': f1}
 
            if args.metric_learning:
                recall_1, nmi = logger.evaluator.metric_performance()
                print ('recall@1: {}, nmi: {}'.format(recall_1, nmi))
                result_dict = {'auc': auc, 'apr': apr, 'acc': acc, 'f1': f1, 'recall@1': recall_1}

            auc_list = [auc, auc_bestauc, auc_bestloss]
            if np.min(auc_list) == auc:
                auc_list = logger.evaluator.auc_list
                apr_list = logger.evaluator.apr_list
            elif np.min(auc_list) == auc_bestauc:
                auc_list = evaluator_bestauc.auc_list
                apr_list = evaluator_bestauc.apr_list
            else:
                auc_list = evaluator_bestloss.auc_list
                apr_list = evaluator_bestloss.apr_list

        elif args.train_mode == 'regression':
            loss, r, pval, (loss_lower, loss_upper), (r_lower, r_upper), (pval_lower, pval_upper) = logger.evaluator.performance_metric()
            loss_bestloss, r_bestloss, pval_bestloss, (loss_bestlower, loss_bestupper), (r_bestlower, r_bestupper), (pval_bestlower, pval_bestupper) = evaluator_bestloss.performance_metric()
            print ('loss: {} ({}, {}), pearsonr: {} ({}, {}), pval {} ({}, {})'.format(loss, loss_lower, loss_upper, r, r_lower, r_upper, pval, pval_lower, pval_upper))
            ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

            if args.last:
                torch.save(ckpt, os.path.join(args.dir_result, args.name, 'reg_last_last.pth')) # for trend prediction
            else:
                torch.save(ckpt, os.path.join(args.dir_result, args.name, 'reg_last_{}.pth'.format(str(args.load_epoch)))) # for trend prediction

            print('==========================Best Loss==========================')
            print('best loss: {} epoch'.format(logger.bestloss_iter))
            print ('loss: {} ({}, {}), pearsonr: {} ({}, {}), pval {} ({}, {})'.format(loss_bestloss, loss_bestlower, loss_bestupper, r_bestloss, r_bestlower, r_bestupper, pval_bestloss, pval_bestlower, pval_bestupper))
            bestloss_ckpt = {'model': model_bestloss.state_dict()}

            if args.last:
                torch.save(bestloss_ckpt, os.path.join(args.dir_result, args.name, 'reg_bestloss_last.pth')) # for trend prediction
            else:
                torch.save(bestloss_ckpt, os.path.join(args.dir_result, args.name, 'reg_bestloss_{}.pth'.format(str(args.load_epoch)))) # for trend prediction
            result_dict = {'rmse': loss}

            if loss < loss_bestloss:
                rmse_list = logger.evaluator.loss_list
            else:
                rmse_list = evaluator_bestloss.loss_list

    torch.save(result_dict, result_ckpt)
    y_pred, y_true = logger.evaluator.return_pred()
    return y_pred, y_true, auc_list, apr_list, rmse_list
    
y_pred, y_true, auc_list, apr_list, rmse_list = downstream_test(test_loader, args, device, model, model_bestauc, model_bestloss, logger)

if args.sex_perf:
    male_logger = Logger(args, model=model)
    female_logger = Logger(args, model=model)
    print('==========================Female ==========================')
    female_pred, female_true, female_auc_list, female_apr_list, female_rmse_list = downstream_test(female_loader, args, device, model, model_bestauc, model_bestloss, female_logger)
    print('==========================Male ==========================')
    male_pred, male_true, male_auc_list, male_apr_list, male_rmse_list = downstream_test(male_loader, args, device, model, model_bestauc, model_bestloss, male_logger)

    if args.train_mode == 'binary_class':
        # import ipdb; ipdb.set_trace()
        kruskal_stat, kruskal_p_value, t_stat, t_p_value = stat_testing(male_auc_list, female_auc_list)
        print("Kruskal-Wallis test for AUC: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nIndependent t-test for AUC: statistics {} pvalue {}".format(t_stat, t_p_value))

        kruskal_stat, kruskal_p_value, t_stat, t_p_value = stat_testing(male_apr_list, female_apr_list)
        print("Kruskal-Wallis test for APR: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nIndependent t-test for APR: statistics {} pvalue {}".format(t_stat, t_p_value))
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'male_auc_')+'.npy'), male_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'male_apr_')+'.npy'), male_apr_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'female_auc_')+'.npy'), female_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'female_apr_')+'.npy'), female_apr_list)

    else:
        kruskal_stat, kruskal_p_value, t_stat, t_p_value = stat_testing(male_rmse_list, female_rmse_list)
        print("Kruskal-Wallis test for RMSE: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nIndependent t-test for RMSE: statistics {} pvalue {}".format(t_stat, t_p_value))
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'male_rmse_')+'.npy'), male_rmse_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'female_rmse_')+'.npy'), female_rmse_list)
    np.save(os.path.join(args.dir_result, args.name, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.dir_result, args.name, 'y_true.npy'), y_true)
    np.save(os.path.join(args.dir_result, args.name, 'male_pred.npy'), male_pred)
    np.save(os.path.join(args.dir_result, args.name, 'male_true.npy'), male_true)
    np.save(os.path.join(args.dir_result, args.name, 'female_pred.npy'), female_pred)
    np.save(os.path.join(args.dir_result, args.name, 'female_true.npy'), female_true)

if args.age_perf:
    age1_logger = Logger(args, model=model)
    age2_logger = Logger(args, model=model)
    age3_logger = Logger(args, model=model)
    age4_logger = Logger(args, model=model)

    print('========================== Age1 (18 <= age < 35) ==========================')
    age1_pred, age1_true, age1_auc_list, age1_apr_list, age1_rmse_list = downstream_test(age1_loader, args, device, model, model_bestauc, model_bestloss, age1_logger)
    print('========================== Age2 (35 <= age < 50) ==========================')
    age2_pred, age2_true, age2_auc_list, age2_apr_list, age2_rmse_list = downstream_test(age2_loader, args, device, model, model_bestauc, model_bestloss, age2_logger)
    print('========================== Age3 (50 <= age < 75) ==========================')
    age3_pred, age3_true, age3_auc_list, age3_apr_list, age3_rmse_list = downstream_test(age3_loader, args, device, model, model_bestauc, model_bestloss, age3_logger)
    print('========================== Age4 (75 <= age) ==========================')
    age4_pred, age4_true, age4_auc_list, age4_apr_list, age4_rmse_list = downstream_test(age4_loader, args, device, model, model_bestauc, model_bestloss, age4_logger)

    if args.train_mode == 'binary_class':
        kruskal_stat, kruskal_p_value, anova_stat, anova_p_value = stat_testing(age1_auc_list, age2_auc_list, age3_auc_list, age4_auc_list)
        print("Kruskal-Wallis test for AUC: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nIndependent t-test for AUC: statistics {} pvalue {}".format(anova_stat, anova_p_value))

        kruskal_stat, kruskal_p_value, t_stat, t_p_value = stat_testing(age1_apr_list, age2_apr_list, age3_apr_list, age4_apr_list)
        print("Kruskal-Wallis test for APR: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nOne-way ANOVA test for APR: statistics {} pvalue {}".format(anova_stat, anova_p_value))
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age1_auc_')+'.npy'), age1_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age1_apr_')+'.npy'), age1_apr_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age2_auc_')+'.npy'), age2_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age2_apr_')+'.npy'), age2_apr_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age3_auc_')+'.npy'), age3_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age3_apr_')+'.npy'), age3_apr_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age4_auc_')+'.npy'), age4_auc_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age4_apr_')+'.npy'), age4_apr_list)

    else:
        kruskal_stat, kruskal_p_value, anova_stat, anova_p_value = stat_testing(age1_rmse_list, age2_rmse_list, age3_rmse_list, age4_rmse_list)
        print("Kruskal-Wallis test for RMSE: statistics {} pvalue {}".format(kruskal_stat, kruskal_p_value))
        print("\nIndependent t-test for RMSE: statistics {} pvalue {}".format(anova_stat, anova_p_value))
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age1_rmse_')+'.npy'), age1_rmse_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age2_rmse_')+'.npy'), age2_rmse_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age3_rmse_')+'.npy'), age3_rmse_list)
        np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'age4_rmse_')+'.npy'), age4_rmse_list)
    np.save(os.path.join(args.dir_result, args.name, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.dir_result, args.name, 'y_true.npy'), y_true)
    np.save(os.path.join(args.dir_result, args.name, 'age1_pred.npy'), age1_pred)
    np.save(os.path.join(args.dir_result, args.name, 'age1_true.npy'), age1_true)
    np.save(os.path.join(args.dir_result, args.name, 'age2_pred.npy'), age2_pred)
    np.save(os.path.join(args.dir_result, args.name, 'age2_true.npy'), age2_true)
    np.save(os.path.join(args.dir_result, args.name, 'age3_pred.npy'), age3_pred)
    np.save(os.path.join(args.dir_result, args.name, 'age3_true.npy'), age3_true)
    np.save(os.path.join(args.dir_result, args.name, 'age4_pred.npy'), age4_pred)
    np.save(os.path.join(args.dir_result, args.name, 'age4_true.npy'), age4_true)

if args.train_mode == 'classification':
    np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'auc_')+'.npy'), auc_list)
    np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'apr_')+'.npy'), apr_list)
else:
    np.save(os.path.join(args.dir_result, args.name, trend_name.replace('trend_', 'rmse_')+'.npy'), rmse_list)

if args.trend_pred:
    trend_pred(args, model)
