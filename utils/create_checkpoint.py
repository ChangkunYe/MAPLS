import os
import argparse
import ast
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from utils.utils import get_stamp
import time
import logging
import torch
from utils.config_parse import config_setup

parser = argparse.ArgumentParser(description='Long-Tailed Model training')
parser.add_argument('--config', default=None, help='path to config file')
parser.add_argument('--save_path', default=None, type=str, help='model saved path, default ./saved_models')
parser.add_argument('--ft_path', default=None, type=str, help='ImageNet pretrained network path.')


def main():
    if not torch.cuda.is_available():
        raise Exception('GPU is not available reported by torch.cuda.is_available().')
    args = parser.parse_args()

    cfg, _, _ = config_setup(args.config,
                             None,
                             None,
                             update=False)

    state_dict = torch.load(args.ft_path)
    cfg.update(['state_dict', 'model'], state_dict)

    model_stamp = get_stamp(cfg)
    time_stamp = time.strftime('%Y%m%d-%H%M%S')
    log_stamp = cfg.dataset['name'] + '-' + model_stamp + '-' + time_stamp

    cfg.save(args.save_path + log_stamp + '.pth')


if __name__ == '__main__':
    main()
