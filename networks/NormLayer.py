import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features, p=1):
        super().__init__()
        # p is the exponent value in the norm formulation
        self.p = p
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(p=2, dim=0, maxnorm=1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=-1).mm(F.normalize(self.weight.T, p=self.p, dim=0))
        # out = F.normalize(x, p=self.p, dim=-1).mm(F.normalize(self.weight.T, p=self.p, dim=0))
        return out


# Normlayer used in LDAM-DRW https://github.com/kaidic/LDAM-DRW
class NormedLinear_LDAM(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out