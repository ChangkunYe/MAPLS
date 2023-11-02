import torch
import numpy as np


class NoiseMixUp:
    def __init__(self, batch_size, alpha, noise_rate=0.0):
        assert 0 <= noise_rate < 1 and 0 <= alpha < 1
        self.weight = 1 if alpha == 0 else np.random.beta(alpha, alpha)
        if self.weight <= 0.5:
            self.weight = 1 - self.weight
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)
        self.noise_rate = noise_rate

    def augment_input(self, img):
        img = self.weight * img + (1 - self.weight) * img[self.new_idx, :]
        noise = torch.rand_like(img)
        return (1 - self.noise_rate) * img + self.noise_rate * noise

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat(len(labels) // self.batch_size)
        return self.weight * criterion(logits, labels) + (1 - self.weight) * criterion(logits, new_labels)