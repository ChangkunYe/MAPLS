import torch
import numpy as np
import torch.nn.functional as F


class DataAugMethod:
    def __init__(self, config, batch_size, cls_num_list=None, labels=None):
        if config['name'] == 'mixup':
            self.method = MixUp(batch_size, config['alpha'])
        elif config['name'] == 'cutmix':
            self.method = CutMix(batch_size, config['alpha'])
        elif config['name'] == 'unimix':
            assert labels is not None
            assert cls_num_list is not None

            self.method = UniMix(batch_size, config['alpha'], config['tau'], cls_num_list, labels)
        else:
            raise Exception('Unsupported mixup type, expect in [\'mixup\', \'cutmix\'], got %s' % config['name'])

    def augment_input(self, img):
        return self.method.augment_input(img)

    def augment_criterion(self, criterion, logits, labels):
        return self.method.augment_criterion(criterion, logits, labels)


# MixUp, Modified with original code: https://github.com/facebookresearch/mixup-cifar10
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


class NoiseMixUp:
    def __init__(self, batch_size, alpha, noise_rate=0.0):
        assert 0 <= noise_rate < 1 and 0 < alpha < 1
        self.weight = 0 if alpha <= 0 else np.random.beta(alpha, alpha)
        if self.weight >= 0.5:
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


# CutMix, Modified with original code: https://github.com/clovaai/CutMix-PyTorch
class CutMix:
    def __init__(self, batch_size, beta):
        self.weight = np.random.beta(beta, beta)
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)

    def augment_input(self, img):
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape, self.weight)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[self.new_idx, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        self.weight = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))

        return img

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat(len(labels) // self.batch_size)
        return criterion(logits, labels) * self.weight + criterion(logits[self.new_idx], new_labels) * (1. - self.weight)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# UniMix, Modified from Official Code https://github.com/XuZhengzhuo/Prior-LT
class UniMix:
    def __init__(self, batch_size, alpha, tau, cls_num_list, labels):
        self.batch_size = batch_size
        self.new_idx = unimix_sampler(batch_size, labels, cls_num_list, tau)
        self.weights = unimix_factor(labels, labels[self.new_idx], cls_num_list, alpha)

    def augment_input(self, img):
        return img * self.weights[(...,) + (None,) * 3] + \
               img[self.new_idx] * (1 - self.weights)[(...,) + (None,) * 3]

    def augment_criterion(self, criterion, logits, labels):
        ensemble_num = len(labels) // self.batch_size
        new_labels = labels[self.new_idx].repeat(ensemble_num)
        return criterion(logits, labels) * self.weights.repeat(ensemble_num) + \
               criterion(logits, new_labels) * (1 - self.weights.repeat(ensemble_num))


def unimix_sampler(batch_size, labels, cls_num_list, tau):
    idx = np.linspace(0, batch_size - 1, batch_size, dtype=int)
    cls_num = np.array(cls_num_list)
    idx_prob = cls_num[labels]
    idx_prob = np.power(idx_prob, tau, dtype=float)
    idx_prob = idx_prob / np.sum(idx_prob)
    idx = np.random.choice(idx, batch_size, p=idx_prob)
    # idx = torch.Tensor(idx).type(torch.LongTensor)
    return idx


def unimix_factor(labels_1, labels_2, cls_num_list, alpha):
    cls_num_list = np.array(cls_num_list)
    n_i = cls_num_list[labels_1]
    n_j = cls_num_list[labels_2]
    lam = n_j / (n_i + n_j)
    lam = [np.random.beta(alpha, alpha) + t for t in lam]
    lam = np.array([t - 1 if t > 1 else t for t in lam])
    return torch.Tensor(lam).cuda()
