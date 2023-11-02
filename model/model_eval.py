import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.data_aug.MixUp import MixUp
from model.data_aug.NoiseMixUp import NoiseMixUp
from utils.utils import normalized
import logging


def model_eval(model, dataset, ls_est=False):
    model.eval()
    if ls_est:
        logging.info('Augmenting Input Image to improve Label Shift Estimation.')

    labels = []
    probs = []
    with torch.no_grad():
        for i, (x, y, path) in enumerate(dataset):
            batch_size = len(y.numpy())
            labels.extend(list(y.numpy()))
            img = x.cuda()

            # Adding noise to the input if required for label shift estimation purpose
            # if ls_est:
            #    data_aug = NoiseMixUp(batch_size, 0.1, 0.1)
            #    img = data_aug.augment_input(img)

            logit = model(img).detach()
            prob = F.softmax(logit, dim=-1)
            probs.extend(list(prob.cpu().numpy()))

    return np.array(probs), labels

