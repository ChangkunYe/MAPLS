import os
import argparse
import ast
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from data_loader.data_loader_wrapper import data_loader_wrapper
from trainer import trainer
from test import test
from tester import tester
from utils.utils import get_stamp
import time
import logging
import torch
from utils.config_parse import config_setup

parser = argparse.ArgumentParser(description='Long-Tailed Model training')
parser.add_argument('--config', default=None, help='path to config file')
parser.add_argument('--is_test', default=False, type=bool, help='if True please give path to .checkpoint in model path')
parser.add_argument('--datapath', default='./data/', type=str, help='dataset path')
parser.add_argument('--savepath', default='./saved_models/', type=str, help='model saved path, default ./saved_models')
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--ft_path', default=None, type=str, help='ImageNet pretrained network path.')
parser.add_argument('--test_config', default=None, type=str, help='imbalance testset config path')


def main():
    if not torch.cuda.is_available():
        raise Exception('GPU is not available reported by torch.cuda.is_available().')
    args = parser.parse_args()

    # ---------------------Test Set Config Parsing----------------------------#
    if args.test_config is None:
        test_cfg = None
    else:
        print('--test_config is given, use imbalance test and valiation set setup given in the config.')
        with open(args.test_config, 'r') as f:
            test_cfg = ast.literal_eval(f.read().replace(' ', '').replace('\n', ''))

    if args.is_test:
        # ------------Direct Test the model if "--is_test True"---------------#
        _ = test(datapath=args.datapath,
                 modelpath=args.checkpoint,
                 test_cfg=test_cfg)
    else:
        # ---------------------------Config Parsing---------------------------#
        # Load config and checkpoint from arg paths, update checkpoint with config if set 'update=True'
        cfg, finish = config_setup(args.config,
                                   args.checkpoint,
                                   args.datapath,
                                   update=False)

        # If the info in checkpoint and config indicate training has already finished, jump to model test
        if finish:
            print('Model is fully trained according to checkpoint info. Directly testing the model.')
            print('It is recommend to set \"--is_test True\" to directly test the model.')
            _ = test(args.datapath,
                     modelpath=args.checkpoint,
                     test_cfg=test_cfg)
            return 0

        # -----------------------------Logger Setup---------------------------#
        # Generate Different log on each calling of main().
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        # Create time stamp of current run
        time_stamp = time.strftime('%Y%m%d-%H%M%S')

        # Create imbalance factor stamp of current run
        imb_factor = cfg.dataset['imb_factor']
        imb_stamp = str(int(1 / imb_factor) if type(imb_factor) == float else str(imb_factor))

        # Create model structure stamp of current run (model name, loss name, mixup .etc) see utils.utils.get_stamp
        model_stamp = get_stamp(cfg)

        # Create log name based on the 3 stamps
        log_stamp = cfg.dataset['name'] + '-' + imb_stamp + '-' + model_stamp + '-' + time_stamp
        logname = './log/run-' + log_stamp + '.log'

        # Setup logger, start logging
        logging.basicConfig(filename=logname, level=logging.INFO)
        logging.info('Random Seed: %i' % (torch.random.seed()))
        if test_cfg is not None:
            logging.info('Using imbalanced test set config: ' + str(test_cfg))
        logging.info('Now training the model on dataset: ' + cfg.dataset['name'])

        # --------------------logging the config detail---------------#
        if torch.cuda.device_count() == 1:
            logging.info(
                'Only 1 visible GPU available in the system, set ["dataset"]["num_workers"] = 4 to reduce load.')
            cfg.update(['dataset', 'num_workers'], 4)
        logging.info('Hyperparameters: ' + cfg.print())

        # -----------------------Model Training-----------------------#
        logging.info('Training model.')
        time_start = time.time()
        # --------------Loading the dataset, create dataloader--------#
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset, test_cfg)
        # update per-class image number in the dataset
        cfg.update(['train_info', 'class_num_list'], dset_info['per_class_img_num'])
        # update model save path with log_stamp defined in logger config section
        cfg.update(['checkpoint', 'save_path'], args.savepath + log_stamp + '.checkpoint')
        # ----------------------------Train Model---------------------#
        trainer(train_set, test_set, cfg, resume=cfg.resume, log_stamp=log_stamp)

        # ---------------------------Test the Model---------------------------#
        logging.info('Test the model')
        logging.info('Test with chekpoint loading at: %s' % cfg.checkpoint['save_path'])
        acc = tester(test_set, train_set, val_set,
                     torch.load(cfg.checkpoint['save_path']), log_stamp)


if __name__ == '__main__':
    main()