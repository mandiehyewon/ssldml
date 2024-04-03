import torch
import torch.nn as nn
import torch.nn.functional as F

class Finetune_Classifier(nn.Module):
    def __init__(self, args, dml_model):
        super(Finetune_Classifier, self).__init__()
        self.args = args
        self.dml_model = dml_model

        self.linear = nn.Linear(args.embedding_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.num_classes)
        self.batchnorm = nn.BatchNorm1d(num_features=args.hidden_dim)

    def forward(self, x, get_embedding=False):
        embed = self.dml_model(x)

        if get_embedding:
            return embed
        
        x = F.relu(self.batchnorm(self.linear(embed)))
        x = nn.Dropout(self.args.dropout)(x)
        out = self.linear2(x)
        
        return out