import numpy as np
import torch


class MixUp:
    def __init__(self, batch_size, alpha):
        self.weight = np.random.beta(alpha, alpha)
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)

    def augment_input(self, img):
        return self.weight * img + (1 - self.weight) * img[self.new_idx, :]

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat(len(labels) // self.batch_size)
        return self.weight * criterion(logits, labels) + (1 - self.weight) * criterion(logits, new_labels)