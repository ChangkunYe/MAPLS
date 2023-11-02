import numpy as np
from label_shift.label_shift_est import get_label_shift_ratio
import logging


def get_cls_weight(cfg, epoch, milestones, train_cls_num_list,
                   train_probs=None, train_labels=None, val_probs=None):
    if cfg['mode'] in ['Normal', 'DRW', 'Reweight']:
        # -----------------------DRW and Resampling setup ----------------------#
        # Original code: https://github.com/kaidic/LDAM-DRW
        if cfg['mode'] == 'Normal':
            per_cls_weights = None
        elif cfg['mode'] == 'Reweight':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, train_cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_cls_num_list)
            per_cls_weights = per_cls_weights
        elif cfg['mode'] == 'DRW':
            idx = 0 if epoch <= milestones[-1] else 1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], train_cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_cls_num_list)

    elif cfg['mode'] in ['BBSERW', 'MLLSRW', 'RLLSRW']:
        assert cfg['mode'][:-2] == cfg['label_shift']['name']
        if not all(x is not None for x in [train_probs, train_labels, val_probs]):
            logging.info('[warning] Training requires label-shift reweighting,'
                         'but train_probs, train_labels or val_probs not given')

        if epoch <= cfg['label_shift']['start_epoch']:
            per_cls_weights = None
        else:
            per_cls_weights = get_label_shift_ratio(train_probs, train_labels, val_probs, train_cls_num_list,
                                                    py_mode=cfg['label_shift']['py_mode'],
                                                    qy_mode=cfg['label_shift']['qy_mode'],
                                                    qy_method=cfg['label_shift']['name'],
                                                    max_iter=cfg['label_shift']['max_iter'])

            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_cls_num_list)
    else:
        raise Exception('Unsupported train_mode, expect in '
                        '["Normal", "Reweight", "DRW", "BBSERW", "RLLSRW", "MLLSRW"]')

    return per_cls_weights
