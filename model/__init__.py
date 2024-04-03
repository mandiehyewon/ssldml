import os
import importlib

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from model.resnet import resnet18, resnet34
from model.dtw_resnet import resnet18_dtw
from model.classifier import Classifier
from model.finetune_cnn import Finetune_Classifier

def get_model(args, device=None, valid=False, dtw=False, finetuning=False, dml_model=None):
    if valid:
        model = Classifier(args=args) #two layer fc classifier (with BN, ReLU, Dropout)
        
    elif finetuning:        
        # Load model for finetuning
        model = Finetune_Classifier(args, dml_model)
        print('model for finetuning')
        
    else:
        if args.model == "cnn":
            if args.dtw or dtw:
                model = resnet18_dtw(pretrained=args.pretrain, args=args)
            else:
                model = resnet18(pretrained=args.pretrain, args=args)

        elif args.model == "cnn_prev":
            model = models.resnet18(pretrained=False).to(device)
            model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

        else:
            model_module = importlib.import_module("model." + args.model)
            model_class = getattr(model_module, args.model.upper())
            model = model_class(args)

    model = model.to(device)

    return model

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, args):
        super(TFC, self).__init__()
        self.encoder_t = resnet18(pretrained=False, args=args)

        self.projector_t = nn.Sequential(
            nn.Linear(args.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.encoder_f = resnet18(pretrained=False, args=args)

        self.projector_f = nn.Sequential(
            nn.Linear(args.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
    
"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, args):
        super(target_classifier, self).__init__()
        self.args = args
        self.batchnorm = nn.BatchNorm1d(num_features=args.hidden_dim)
        self.linear = nn.Linear(2*args.embedding_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.num_classes)

    def forward(self, emb):
        x = F.relu(self.batchnorm(self.linear(emb)))
        x = nn.Dropout(self.args.dropout)(x)
        out = self.linear2(x)
        return out