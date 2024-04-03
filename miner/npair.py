import os
import numpy as np
import torch


class BatchMiner():
    def __init__(self, args):
        self.args          = args
        self.name         = 'npair'

    def __call__(self, batch, labels, train_x=None, step=0):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        anchors, positives, negatives = [],[],[]

        for i in range(len(batch)):
            anchor = i
            pos    = labels == labels[anchor]

            if np.sum(pos)>1:
                anchors.append(anchor)
                avail_positive = np.where(pos)[0]
                avail_positive = avail_positive[avail_positive!=anchor]
                positive       = np.random.choice(avail_positive)
                positives.append(positive)

        negatives = []
        for anchor,positive in zip(anchors, positives):
            neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive] and labels[i] != labels[anchor]]
            negative_set = np.random.choice(np.arange(len(batch))[neg_idxs])
            negatives.append(negative_set)

        if self.args.loss == 'angular':
            return anchors, positives, negatives

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if self.args.save_triplet:
            if self.args.train_mode == 'binary_class':
                save_dir = self.args.dir_binclasstriplets
            else:
                save_dir = self.args.dir_regtriplets
            torch.save(train_x.detach().cpu(), os.path.join(save_dir, str(step)+'_x.pth'))
            np.save(os.path.join(save_dir, str(step)+'.npy'), np.stack((anchors, positives, negatives)))

        return sampled_triplets