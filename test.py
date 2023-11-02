import argparse
import numpy as np
import time
import logging
import torch
from data_loader.data_loader_wrapper import data_loader_wrapper
from tester import tester
from utils.utils import logging_setup


def test(datapath, modelpath, test_cfg=None):
    # ---------------------------Load Saved Model---------------------------#
    config = dict(torch.load(modelpath))

    # -------------------------Logging Config-------------------------------#
    time_stamp = time.strftime('%Y%m%d-%H%M%S')
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    log_stamp = logging_setup(modelpath, time_stamp, config, test_cfg)

    if test_cfg is not None:
        logging.info('Using imbalanced test set config: ' + str(test_cfg))
    # logging.info('Random Seed: %i' % (torch.random.seed()))

    # ----------------Loading the dataset, create dataloader----------------#
    config['dataset']['path'] = datapath + config['dataset']['name']
    if torch.cuda.device_count() == 1:
        logging.info('Only 1 visible GPU available in the system, set ["dataset"]["num_workers"] = 4 to reduce load.')
        config['dataset']['num_workers'] = 4

    train_set, val_set, test_set, dset_info = data_loader_wrapper(config['dataset'], test_cfg)
    config['train_info']['class_num_list'] = dset_info['per_class_img_num']

    # -------------------------Test the Model-------------------------------#
    logging.info('Test performance on test set.')
    acc = tester(test_set, train_set, test_set, config, log_stamp)
    # logging.info('Test performance on train set.')
    # acc = tester(train_set, train_set, val_set, config, log_stamp + '-trainset', analyze=False)

    return acc




