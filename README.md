# Deep Metric Learning for the Hemodynamics Inference with Electrocardiogram Signals

This repository contains a code implementation and pretrained models of [Deep Metric Learning for the Hemodynamics Inference with Electrocardiogram Signals](https://proceedings.mlr.press/v219/jeong23a/jeong23a.pdf) (Jeong et al., MLHC 2023)

## Environment Setting
To begin, install conda, then configure the environment using these steps:
```
conda env create --name ssldml -f env.yml
conda activate ssldml
```

## Organization
Due to restrictions on sharing the data, unfortunately currently we're unable to share the MGB ECG dataset. However, you can modify the code designed for the MGB dataset to other public source ECGs. Important folders within the codebase include:
    .
    ├── criteria                                        # codes for metric loss
    ├── data                                            # data classes, dataloader classes
    ├── models                                          # Models
    ├── miner                                           # Mine Triplets based on distance (DTW, Euclidean), Label, etc.
    ├── utils                                           # Tools and utilities
    └── README.md


# Baselines
Supervised baseline with ResNet18 model for the binary classification of PCWP
```
python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp_bin_resnet18
```
To test the performance of a model for PCWP binary classification, you can run as below. You can change `--last` to `--best-auc` or `--best-loss`.
```
python test.py --train-mode binary_class --model cnn --label pcwp --name pcwp_bin_resnet18 --last
python test.py --name pcwp15_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp
```

## Unsupervised Metric Learning
```
python train_unl_metric.py --model cnn --metric-learning --unlabeled --loss triplet --batch-mining miner_dtw --data-sampler random --embedding-dim 128 --batch-size 64 --epochs 1 --name unl_batch64
```

## Supervised Metric Learning
For supervised metric learning, you can run:
```
python train_metric.py --train-mode regression --class-loss --model cnn --label pcwp --metric-learning --embedding-dim 128 --alpha 3.0 --name sup_bin_metric_epc1000_embed128_alpha3.0
```
To evaluate the metric embedding with logistic regression,
```
python train_metric.py --train-mode regression --model cnn --label pcwp --metric-learning --embedding-dim 128 --test-logistic --name pcwp_metric_bin_resnet18
```
To evaluate the embedding without metric learning with logistic regression,
```
python train_metric.py --train-mode regression --model cnn --label pcwp --metric-learning --metric-freeze --embedding-dim 128 --test-logistic --name pcwp_metric_bin_resnet18
```

You can also apply same setting for binary classification:

For supervised metric learning, you can run:
```
python train_metric.py --train-mode binary_class --model cnn --label pcwp --metric-learning --embedding-dim 128 --name pcwp_metric_bin_resnet18
```
