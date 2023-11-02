import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, weight, scale=1):
        super(CELoss, self).__init__()
        self.weight = weight
        self.s = scale

    def forward(self, logits, targets):
        return F.cross_entropy(self.s * logits, targets, reduction='none', weight=self.weight)