import torch
import torch.nn as nn
import torch.nn.functional as F

# LDAM and Focal loss from https://github.com/kaidic/LDAM-DRW
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0., scale=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.s = scale

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input * self.s, target, reduction='none', weight=self.weight), self.gamma)