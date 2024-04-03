import os
import random
import numpy as np
import torch


class BatchMiner():
    def __init__(self, args):
        self.args          = args
        self.name         = 'labelbased'
        if self.args.normalize_label:
            self.pcwp_train = np.load("./stores/train_info.npy")
            self.pcwp_mean = self.pcwp_train[0]
            self.pcwp_std = self.pcwp_train[1]
            
    def __call__(self, batch, labels, train_x=None, step=0):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        anchors, positives, negatives = [],[],[]

        for i in range(len(batch)):
            anchor   = i
            diff = abs(labels - labels[anchor])
            negatives.append(np.argmax(diff))
            diff[anchor] = np.inf
            anchors.append(anchor)
            positives.append(np.argmin(diff))

        assert len(anchors) == len(positives) == len(negatives)

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        return sampled_triplets