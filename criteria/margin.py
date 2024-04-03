import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

"""================================================================================================="""
# ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_argsIM      = True
REQUIRES_OPTIM      = False
REQUIRES_DEVICE      = False

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, args, batchminer):
        super(Criterion, self).__init__()
        self.n_classes          = args.num_classes
        self.args               = args
        self.margin             = args.loss_margin_margin
        self.nu                 = args.loss_margin_nu
        self.beta_constant      = args.loss_margin_beta_constant
        self.beta_val           = args.loss_margin_beta

        if args.loss_margin_beta_constant:
            self.beta = args.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(args.num_classes+1)*args.loss_margin_beta)

        self.batchminer = batchminer

        self.name  = 'margin'

        self.lr    = args.loss_margin_beta_lr

        ####
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_argsIM      = REQUIRES_argsIM

    def forward(self, batch, labels, step, train_x = None,**kwargs):
        sampled_triplets = self.batchminer(batch, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]].to(torch.int)] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss = torch.sum(pos_loss+neg_loss)
            else:
                loss = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: 
                beta_regularization_loss = torch.sum(beta)
                loss += self.nu * beta_regularisation_loss.to(torch.float).to(d_ap.device)
        else:
            loss = torch.tensor(0.).to(torch.float).to(batch.device)

        return loss