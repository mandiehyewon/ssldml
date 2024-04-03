import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch

from config import args
from data import get_data
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.utils import set_devices, logit2prob, scatterplot

'''
CUDA_VISIBLE_DEVICES={} python test.py --name {name} --model {cnn} --
CUDA_VISIBLE_DEVICES=1 python test.py --name sup_bin --model cnn --best-auc --train-mode binary_class --supervised --label pcwp --pcwp-th 18 --sex-perf --bootstrap --reset
CUDA_VISIBLE_DEVICES=1 python test.py --name sup_bin --model cnn --best-auc --train-mode binary_class --supervised --label pcwp --pcwp-th 18 --age-perf --bootstrap --reset
'''
# Get Dataloader, Model
name = args.name
if args.sex_perf:
    train_loader, val_loader, test_loader, male_loader, female_loader = get_data(args)
elif args.age_perf:
    train_loader, val_loader, test_loader, age1_loader, age2_loader, age3_loader, age4_loader = get_data(args)
else:
    train_loader, val_loader, test_loader = get_data(args)

device = set_devices(args)

model = get_model(args, device=device)
evaluator = Evaluator(args)
criterion = get_loss(args)

# Check if result exists
result_ckpt = os.path.join(args.dir_result, name, 'test_result.pth')
if (not args.reset) and os.path.exists(result_ckpt):
    print('this experiment has tested before.')
    sys.exit()

# Check if checkpoint exists
if args.last:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/last.pth')
elif args.best_auc:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/bestauc.pth')
elif args.best_loss:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/bestloss.pth')
elif args.load_epoch is not None:
    print('testing from the epoch with epoch {}'.format(args.load_epoch))
    ckpt_path = os.path.join(args.dir_result, args.name, 'ckpts/{}.pth'.format(args.load_epoch))


if not os.path.exists(ckpt_path):
    print("invalid checkpoint path : {}".format(ckpt_path))

# Load checkpoint, model
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt['model']
model.load_state_dict(state)
model.eval()
print('loaded model'+args.name)

evaluator.reset()
if args.plot_prob:
    prob = []
    label = []

if args.sex_perf:
    male_evaluator = Evaluator(args)
    female_evaluator = Evaluator(args)
elif args.age_perf:
    age1_evaluator = Evaluator(args)
    age2_evaluator = Evaluator(args)
    age3_evaluator = Evaluator(args)
    age4_evaluator = Evaluator(args)

if args.normalize_label:
    pcwp_train = np.load("./stores/train_info.npy")
    pcwp_mean = pcwp_train[0]
    pcwp_std = pcwp_train[1]

def test_loop(test_loader, args, device, model, evaluator):
    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            if args.plot_prob: #and (args.train_mode == 'binary_class'):
                test_x, test_y, pressure = test_batch
            else:
                test_x, test_y = test_batch
            test_x, test_y = test_x.to(device), test_y.to(device)
            logits = model(test_x)

            if args.normalize_label: # for normalized label for regression task
                logits = logits*pcwp_std+pcwp_mean
                test_y = test_y*pcwp_std+pcwp_mean

            loss = criterion(logits.float(), test_y.unsqueeze(1).float())
            evaluator.add_batch(test_y.cpu(), logits.cpu(), loss, test=True)

            if args.plot_prob:
                if args.train_mode == 'regression':
                    prob_np = np.array(logits.cpu().float())
                else:
                    prob_np = np.apply_along_axis(logit2prob, 0, np.array(logits.cpu()))
                    
                if type(prob) == list:
                    prob = prob_np
                    label = np.array(pressure)
                else:
                    prob = np.concatenate((prob, prob_np))
                    label = np.concatenate((label, np.array(pressure)))
            
        if args.train_mode == 'binary_class':
            if args.bootstrap:
                f1, (f1_lower, f1_upper), auc, (auc_lower, auc_upper), apr, (apr_lower, apr_upper), acc, (acc_lower, acc_upper) = evaluator.performance_metric()
                print ('auc: {} ({}, {}) \napr: {} ({}, {}) \nacc: {} ({}, {}) \nf1: {} ({}, {})'.format(auc, auc_lower, auc_upper, apr, apr_lower, apr_upper, acc, acc_lower, acc_upper, f1, f1_lower, f1_upper))
            else:
                f1, auc, apr, acc = evaluator.performance_metric()
                print ('auc: {}\n apr: {} \n acc: {} \n f1: {}'.format(auc, apr, acc, f1))
            result_dict = {'auc': auc, 'apr': apr, 'acc': acc, 'f1': f1}

        elif args.train_mode == 'regression':
            if args.bootstrap:
                loss, r, pval, (loss_lower, loss_upper), (r_lower, r_upper), (pval_lower, pval_upper) = evaluator.performance_metric()
                print ('loss: {} ({}, {}) \n pearsonr: {} ({}, {}) \n pval : {} ({}, {})'.format(loss, loss_lower, loss_upper, r, r_lower, r_upper, pval, pval_lower, pval_upper))
            else:
                loss, r, pval = evaluator.performance_metric()
                print ('loss: {}\n pearsonr: {}\n pval : {}'.format(loss, r, pval))
            result_dict = {'rmse': loss}

    if args.plot_prob:
        scatterplot(args, label, prob)

    torch.save(result_dict, result_ckpt)
    y_pred, y_true = evaluator.return_pred()
    return y_pred, y_true

y_pred, y_true = test_loop(test_loader, args, device, model, evaluator)

if args.sex_perf:
    print('==========================Female ==========================')
    female_pred, female_true =test_loop(female_loader, args, device, model, female_evaluator)
    print('==========================Male ==========================')
    male_pred, male_true = test_loop(male_loader, args, device, model, male_evaluator)

    np.save(os.path.join(args.dir_result, args.name, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.dir_result, args.name, 'y_true.npy'), y_true)
    np.save(os.path.join(args.dir_result, args.name, 'male_pred.npy'), male_pred)
    np.save(os.path.join(args.dir_result, args.name, 'male_true.npy'), male_true)
    np.save(os.path.join(args.dir_result, args.name, 'female_pred.npy'), female_pred)
    np.save(os.path.join(args.dir_result, args.name, 'female_true.npy'), female_true)

elif args.age_perf:
    print('========================== Age1 (18 <= age < 35) ==========================')
    age1_pred, age1_true =test_loop(age1_loader, args, device, model, age1_evaluator)
    print('========================== Age2 (35 <= age < 50) ==========================')
    age2_pred, age2_true = test_loop(age2_loader, args, device, model, age2_evaluator)
    print('========================== Age3 (50 <= age < 75) ==========================')
    age3_pred, age3_true = test_loop(age3_loader, args, device, model, age3_evaluator)
    print('========================== Age4 (75 <= age) ==========================')
    age4_pred, age4_true = test_loop(age4_loader, args, device, model, age4_evaluator)

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