import torch
import numpy as np
import torch.nn.functional as F

class DataAugMethod:
    def __init__(self, config, batch_size):
        if config['name'] == 'mixup':
            from model.data_aug.MixUp import MixUp
            self.method = MixUp(batch_size, config['alpha'])
        elif config['name'] == 'nmixup':
            from model.data_aug.NoiseMixUp import NoiseMixUp
            self.method = NoiseMixUp(batch_size, config['alpha'], config['noise_rate'])
        else:
            raise Exception('Unsupported mixup type, expect in [\'mixup\', \'cutmix\'], got %s' % config['name'])

    def augment_input(self, img):
        return self.method.augment_input(img)

    def augment_criterion(self, criterion, logits, labels):
        return self.method.augment_criterion(criterion, logits, labels)