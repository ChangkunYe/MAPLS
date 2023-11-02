import numpy as np
import torch
import torch.nn as nn
import copy
import json

def save_model(savepath, params, model, optimizer, lr_scheduler=None):
    checkpoint = params
    if not isinstance(model, nn.DataParallel):
        checkpoint['state_dict'] = model.state_dict()
    else:
        checkpoint['state_dict'] = model.module.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['scheduler_state_dict'] = lr_scheduler.state_dict()

    torch.save(checkpoint, savepath)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismtach found at', key_item_1[0])
            else:
                print(key_item_1[0], key_item_2[0])
    if models_differ == 0:
        print('Models match perfectly! :)')


def get_stamp(cfg):
    lrs = cfg.lr_scheduler['name'][0] if 'lr_scheduler' in cfg.keys else 'n'
    stamp = cfg.model['name'] + '-' + \
            lrs + \
            cfg.loss['name'][0] + \
            cfg.train_info['mode'][0] + \
            cfg.train_info['data_aug']['name'][0]

    return stamp


def logging_setup(path, time_stamp, config, test_cfg=None):
    import logging
    model_stamp = '-'.join(path.split('/')[-1].split('-')[:-2])
    log_stamp = model_stamp + '-' + time_stamp
    log_stamp = log_stamp + '-None' if test_cfg is None else log_stamp + '-' + test_cfg['mode']
    logname = './log/test-' + log_stamp + '.log'
    logging.basicConfig(filename=logname, level=logging.INFO)
    # Generate print version of config (without [\'state_dict\'], [\'train_info\'][\'class_num_list\'])
    cfg_print = copy.deepcopy({k: config[k] for k in set(list(config.keys())) - set(['state_dict'])})
    cfg_print['train_info'].pop('class_num_list', None)
    logging.info('Hyperparameters: ' + json.dumps(cfg_print, indent=4, default=default))

    return log_stamp


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.round(4).tolist()
    elif isinstance(obj, np.float32):
        return round(float(obj), 4)
    raise TypeError('Not serializable')