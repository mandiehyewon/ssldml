import torch
import torch.nn as nn
import torch.nn.functional as F


class DTW_CLASSIFIER(nn.Module):
    def __init__(self, args):
        super(DTW_CLASSIFIER, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.linear1 = nn.Linear(args.hidden_dim, args.embedding_dim) #128, 256
        self.linear2 = nn.Linear(args.embedding_dim, args.num_classes)
        self.batchnorm = nn.BatchNorm1d(num_features=args.input_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=args.hidden_dim)

    def forward(self, x1, x2):
        x1_conv = F.relu(self.batchnorm(self.conv1(x1)))
        x2_conv = F.relu(self.batchnorm(self.conv1(x2)))
        print(x1_conv.size(), x2_conv.size())
        import ipdb; ipdb.set_trace()
        x = torch.cat((x1_conv.view(x1_conv.size(0),-1), 
                      x2_conv.view(x2_conv.size(0), -1)), dim=0)
        x = F.relu(self.batchnorm(self.linear1(x)))
        x = nn.Dropout(self.args.dropout)(x)
        out = self.linear2(x)
        
        return out