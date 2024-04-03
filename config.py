import os
import yaml
import argparse

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# General Parameters
parser.add_argument("--cpu", default=False, action="store_true")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--reset", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=926)

# Training Parameters
parser.add_argument("--supervised", default=False, action="store_true") # supervised baseline
parser.add_argument("--train-mode", type=str, default="regression", choices=["regression", "binary_class"])
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--scheduler", type=str, default="poly", choices=["poly", "cos"])
parser.add_argument('--lr-sch-start', type=int, default=0)
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--decay", type=bool, default=True)
parser.add_argument("--decay-rate", type=int, default=0.1)
parser.add_argument("--decay-iter", type=int, default=56000)

# Metric Learning Parameters
parser.add_argument("--metric-learning", default=False, action="store_true") #metric-learning
parser.add_argument('--loss',            default='margin',      type=str,   help='Training criteria: For supported methods, please check criteria/__init__.py')
parser.add_argument('--batch-miner',    default='npair',    type=str,   help='Batchminer for tuple-based losses: For supported methods, please check batch_mining/__init__.py')
parser.add_argument("--recallk", type=int, default=1, help='recall_at_k')
parser.add_argument("--pretraining", default=False, action="store_true")
parser.add_argument("--finetuning", default=False, action="store_true")
parser.add_argument("--neg-random", default=False, action="store_true", help='random mining of negative samples')
parser.add_argument("--neg-topk", type=int, default=2, help='top k negative samples randomly sample from')

# Data Parameters
parser.add_argument('--data-sampler', default='class_random', type=str, help='How the batch is created. Available options: See datasampler/__init__.py.')
parser.add_argument("--unlabeled", default=False, action="store_true") #unlabeled
parser.add_argument('--unl-size', type=int, default=1000)
parser.add_argument("--limited", default=False, action="store_true") #setting for limited data point (only 1,000 data for downstream finetuning)
parser.add_argument('--data', type=str, default="whole") #mgh, bgw
parser.add_argument("--normalize-label", default=False, action="store_true")  # used to normalize labels (pcwp)
parser.add_argument("--label", type=str, default="pcwp", choices=["pcwp", "age", "sex", "CO", "mPAP"])
parser.add_argument("--pcwp-th", type=int, default=18)
parser.add_argument('--num-classes', type=int, default=1)
parser.add_argument("--ecg-len", type=int, default=2500)
parser.add_argument('--data-idx', type=int, default=1)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--sex-perf", default=False, action="store_true") # sex subgroup performance
parser.add_argument("--age-perf", default=False, action="store_true") # age subgroup performance
parser.add_argument("--drop-last", default=False, action="store_true")

# DTW Parameters
parser.add_argument("--dtw", default=False, action="store_true")
parser.add_argument("--matrix", default=False, action="store_true")
parser.add_argument("--surr-path", type=str, default="dtw_surrogate/dtwsurrogate_cnn_200epc/ckpts/bestloss.pth")

# Model Parameters
parser.add_argument("--model", type=str, default="cnn")  # model name
parser.add_argument("--pretrain", default=False, action="store_true")
parser.add_argument("--load-model", default=False, action="store_true")
parser.add_argument('--load-epoch', type=int, default=None)

# Architecture Parameters
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--input-dim", type=int, default=64)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--embedding-dim", type=int, default=256) #metric-learning

# Loss Parameters
parser.add_argument("--eps", type=float, default=1e-6)  # eps for RMSE
parser.add_argument("--class-loss", default=False, action="store_true", help='Adding CE/MSE loss to metric loss in train loop')
parser.add_argument("--alpha", type=float, default=1.0)  # weight value for triplet loss

# DML Objective Parameters
parser.add_argument('--loss_triplet_margin',       default=0.2,         type=float, help='Margin for Triplet Loss')
parser.add_argument("--save-triplet", default=False, action="store_true")
parser.add_argument("--load-triplet", default=False, action="store_true")

# MarginLoss
parser.add_argument('--loss_margin_margin',       default=0.2,          type=float, help='Triplet margin.')
parser.add_argument('--loss_margin_beta_lr',      default=0.0005,       type=float, help='Learning Rate for learnable class margin parameters in MarginLoss')
parser.add_argument('--loss_margin_beta',         default=1.2,          type=float, help='Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--loss_margin_nu',           default=0,            type=float, help='Regularisation value on betas in Margin Loss. Generally not needed.')
parser.add_argument('--loss_margin_beta_constant', action='store_true',              help='Flag. If set, beta-values are left untrained.')

### Angular Loss
parser.add_argument('--loss_angular_alpha',             default=45, type=float, help='Angular margin in degrees.')
parser.add_argument('--loss_angular_npair_ang_weight',  default=2,  type=float, help='Relative weighting between angular and npair contribution.')
parser.add_argument('--loss_angular_npair_l2',          default=0.005,  type=float, help='L2 weight on NPair (as embeddings are not normalized).')

# Logging Parameters
parser.add_argument("--log-iter", type=int, default=10)
parser.add_argument("--val-iter", type=int, default=10)
parser.add_argument("--save-iter", type=int, default=10)
parser.add_argument("--log-metricloss", default=False, action="store_true")

# Test / Eval Parameters
parser.add_argument("--best-auc", default=False, action="store_true")
parser.add_argument("--best-loss", default=False, action="store_true")
parser.add_argument("--last", default=False, action="store_true")
parser.add_argument("--plot-prob", default=False, action="store_true")
parser.add_argument("--bootstrap", default=False, action="store_true", help='For Bootstrapping')
parser.add_argument("--plot-baseline", default=False, action="store_true")
parser.add_argument("--trend-pred", default=False, action="store_true")

# DTW Calculation Parameters
parser.add_argument('--dist-calc', type=str, default='dtw', choices=['dtw', 'euclidean', 'dtwreal'])
parser.add_argument('--ttv', type=str, default='train', help='false if not calculating value for ttv')
parser.add_argument("--multiprocessing", default=False, action="store_true")

# CLOCS Paramters
parser.add_argument("--nviews", type=int, default=2)
parser.add_argument("--contrastive", default=False, action="store_true")
parser.add_argument('--contrastive-mode', type=str, default='cmsc', choices=['cmsc', 'simclr', 'tfc'])
parser.add_argument("--gaussian", default=False, action="store_true")
parser.add_argument("--flipalongx", default=False, action="store_true")
parser.add_argument("--flipalongy", default=False, action="store_true")

# TFC Parameters
parser.add_argument("--temperature", type=int, default=0.2)
parser.add_argument("--use-cosine-similarity", default=False, action="store_true")
parser.add_argument("--jitter-ratio", type=int, default=0.8)

args = parser.parse_args()

# Dataset Path settings
with open("path_configs.yaml") as f:
    path_configs = yaml.safe_load(f)
    args.dir_root = path_configs['dir_root']
    args.dir_csv = path_configs["dir_csv"]
    args.dir_result = path_configs["dir_result"]
    args.dir_apollo = path_configs["dir_apollo"]
    args.dir_unl = path_configs["dir_unl"]
    args.dir_dtw = path_configs["dir_dtw"]
    args.dir_euc = path_configs["dir_euc"]
    args.dir_dtw_prev = path_configs["dir_dtw_prev"]
    args.dir_dtw_real = path_configs["dir_dtw_real"]
    args.dir_triplets = path_configs["dir_triplets"]
    args.dir_dtwrealtriplets = path_configs["dir_dtwrealtriplets"]
    args.dir_euctriplets = path_configs["dir_euctriplets"]
    args.dir_binclasstriplets = path_configs["dir_binclasstriplets"]
    args.dir_regtriplets = path_configs["dir_regtriplets"]
    args.dir_semihardtriplets = path_configs["dir_semihardtriplets"]
    args.dir_softhardtriplets = path_configs["dir_softhardtriplets"]

# Device Settings
if args.device is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = str(args.device[0])
    for i in range(len(args.device) - 1):
        device += "," + str(args.device[i + 1])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
