# by Changkun Ye (Nov 11, 2022)
# Australian National University
# Email: changkun.ye@anu.edu.au

import numpy as np
import logging
from typing import List
import math


# Imbalanced Test Set Generator-----------------------------------------------#
def create_imb_subset(labels: List[int], order_idx: np.ndarray = None, cfg: dict = None, return_count: bool = False):
    r"""
    Generating Class-Imbalanced Test set according to Configuration cfg
    The imbalance test set is generated with different label shift, including:
    Long-Tailed (Exponential) Shift, Step Shift, Knock Out Shift, Dirichlet Shift, Tweak One Shift.

    Args:
        labels:             targets of the PyTorch dataset object
        order_idx:          Order for each class in reference list, can be computed via np.argsort(reference list)[::-1]
        cfg:                Config file for generating imbalance dataset. Which Requires:
                            cfg['mode'] == "LT":        Long-Tailed subset, requires cfg['imb_factor']
                            cfg['mode'] == "Step":      Step subset, requires cfg['step_num'] and cfg['imb_factor']
                            cfg['mode'] == "KnockOut":  Knock Out subset, requires cfg['knockout_ratio']
                            cfg['mode'] == "Dirichlet": Dirichlet subset, requires cfg['dirichlet_alpha']
                            cfg['mode'] == "TweakOne":  Tweak One subset, requires cfg['tweakone_rho']

                            cfg['order'] == 'normal':   LT/Step sample No. have same descending order w.r.t order_idx
                            cfg['order'] == 'reverse':  LT/Step sample No. have reverse descending order w.r.t order_idx
                            cfg['order'] == 'random':   LT/Step sample No. have random descending order w.r.t order_idx

                            cfg['data_ratio']: int = 1: Reduce per-class sample No. with a ratio in (0, 1]
        return_count:       Return per-class sample number of generated Subset, default False.

    Shapes:
        * Input:
            labels:             N       "targets" of the dataset
            order_idx:          C       Order of each class in reference list
            cfg:                {}      Dictionary of config file

        * Output:
            subset_loc_list:    <=N     Locations of samples in subset

    """
    if cfg is None:
        logging.info('cfg is None, return original labels.')
        return labels
    else:
        np.random.seed()

        # ----------------Get Available Class indexes in dataset--------------#
        label_set, num_list = np.unique(labels, return_counts=True)
        sample_nums = np.unique(num_list)

        if len(sample_nums) == 1:
            logging.info('Uniform dataset, per-cls sample numbers counts is: %s' % str(sample_nums))
        else:
            logging.info('[warning] Dataset is not uniform, with per-class sample num: %s, '
                         'imbalance test generation may not work.' % str(sample_nums))

        # ----------------Generate Referenced Orders--------------------------#
        if order_idx is None:
            order_idx = np.arange(len(label_set))[::-1]
        reorder_idx = np.argsort(order_idx)

        # -------------Group Per-Class sample Locations-----------------------#
        cls_loc_list = group_int_list(labels, label_set)
        img_max = num_list.min()
        # -------------Reduce Subset Total Number if Required-----------------#
        if 'data_ratio' in cfg.keys():
            assert 0 < cfg['data_ratio'] <= 1
            if cfg['data_ratio'] != 1:
                img_max = math.trunc(img_max * cfg['data_ratio'])
                logging.info('Reduce test set samples with ratio given by cfg["reduce"] = %.4f'
                             % cfg['data_ratio'])
                logging.info('Original per-cls sample No. %i, Reduced per-cls sample No. %i'
                             % (num_list.min(), img_max))
        cls_num = len(label_set)

        # ----------------Calculate Per-Class Sample Number-------------------#
        if cfg['mode'] == "Reduce":
            assert 'data_ratio' in cfg.keys()
            cls_num_list = [img_max for _ in range(cls_num)]
        elif cfg['mode'] == "LT":
            cls_num_list = exp_imb_gen(img_max, cls_num, cfg['imb_factor'])
        elif cfg['mode'] == "Step":
            cls_num_list = step_imb_gen(img_max, cls_num, cfg['imb_factor'], cfg['step_num'])
        elif cfg['mode'] == "KnockOut":
            cls_num_list = knockout_imb_gen(img_max, cls_num, cfg['knockout_ratio'])
        elif cfg['mode'] == 'Dirichlet':
            cls_num_list = dirichlet_imb_gen(img_max, cls_num, cfg['dirichlet_alpha'], cfg['total_num'], replace=True)
        elif cfg['mode'] == "TweakOne":
            cls_num_list = tweakone_imb_gen(img_max, cls_num, cfg['tweakone_rho'])
        else:
            raise Exception('Unsupported imb_test mode %s' % cfg['model'])

        # ----------------Change the Order if Required------------------------#
        if cfg['order'] == 'reverse':
            cls_num_list = cls_num_list[::-1]
        elif cfg['order'] == 'random':
            np.random.shuffle(cls_num_list)
        elif cfg['order'] == 'normal':
            pass
        else:
            raise Exception("Unsupported order, expect revese, normal or random, got %s" % cfg['order'])

        cls_num_list = np.array(cls_num_list)[reorder_idx]
        # assert np.argmax(cls_num_list) == np.argmax(train_cls_num_list)

        # ----------------Generate Subset sample locations--------------------#
        # subset_loc_list = []
        #_ = [np.random.shuffle(x) for x in cls_loc_list]
        # _ = [subset_loc_list.extend(list(x[:y])) for x, y in zip(cls_loc_list, cls_num_list)]

        subset_loc_list = []
        for loc, num in zip(cls_loc_list, cls_num_list):
            if num <= len(loc):
                subset_loc_list.extend(np.random.choice(loc, num, replace=False))
            else:
                subset_loc_list.extend(loc)
                subset_loc_list.extend(np.random.choice(loc, num - len(loc), replace=True))
                logging.info('[warning] More sample required than available in this class, '
                             'samples are replicated and will introduce error in estimation')

        logging.info('Subset cls_num_list is: ' + str(cls_num_list))

        if not return_count:
            return subset_loc_list
        else:
            return subset_loc_list, cls_num_list.tolist()


# Long-Tailed Shift Test Set--------------------------------------------------#
def exp_imb_gen(img_max, cls_num, imb_factor):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(math.trunc(num))
    if np.array(img_num_per_cls).min() == 0:
        raise Exception("Imbalance factor too small that 0 sample in tail class")

    return img_num_per_cls[::-1]


# Step Shift Test Set---------------------------------------------------------#
def step_imb_gen(img_max, cls_num, imb_factor, step_num):
    img_min = math.trunc(img_max * imb_factor)

    step_len = math.trunc(cls_num / step_num)
    step_reduce = math.trunc((img_max - img_min) / (step_num - 1))

    img_num_per_cls = np.ones(cls_num, dtype=int) * img_min
    for i in range(step_num - 1):
        img_num_per_cls[i * step_len:(i + 1) * step_len] = math.trunc(img_max - step_reduce * i)

    if img_min == 0:
        raise Exception("Imbalance factor too small that 0 sample in tail class")

    return list(img_num_per_cls)


# Knock Out Shift Test Set----------------------------------------------------#
def knockout_imb_gen(img_max, cls_num, ratio):
    assert 0 < ratio < 1
    np.random.seed()
    knockout_num = math.trunc(img_max * cls_num * ratio)
    label_list = np.arange(cls_num).repeat(img_max)

    cls_idx, removed_num_per_cls = np.unique(np.random.choice(label_list, knockout_num, replace=False),
                                             return_counts=True)

    img_num_per_cls = np.zeros(cls_num, dtype=np.int)
    for i, j in zip(cls_idx, removed_num_per_cls):
        img_num_per_cls[i] = int(img_max - j)

    return list(img_num_per_cls)


# Tweak-One Shift Test Set----------------------------------------------------#
def tweakone_imb_gen(img_max, cls_num, rho, replace: bool = False):
    assert 0 < rho < 1

    rho_min = 1 / ((cls_num - 1) * img_max + 1 )
    rho_max = img_max / (img_max + cls_num - 1)
    if rho < rho_min or rho > rho_max:
        logging.info("Recommended range of rho is: " + str([rho_min, rho_max]))
        raise Exception('Provided Rho cannot generate test set with at lease 1 sample in each class.')

    one_idx = np.random.randint(0, cls_num)
    if rho > 1 / cls_num:
        rest_num = math.trunc((img_max / rho - img_max) / cls_num)
        img_num_per_cls = np.ones(cls_num) * rest_num
        img_num_per_cls[one_idx] = img_max
    else:
        rest_num = img_max
        img_num_per_cls = np.ones(cls_num) * rest_num
        img_num_per_cls[one_idx] = math.trunc(img_max * cls_num * rho / (1 - rho))

    return list(img_num_per_cls)


# Dirichlet Shift Test Set----------------------------------------------------#
def dirichlet_imb_gen(img_max, cls_num, alpha,
                      total_num: int = None, max_iter: int = 50, replace: bool = False):
    sample_num = img_max * cls_num
    if total_num is not None:
        assert total_num <= sample_num
        sample_num = total_num

    if not replace:
        for i in range(max_iter):
            probs = np.random.dirichlet([alpha] * cls_num)
            img_num_per_cls = [math.trunc(x * sample_num) for x in probs]
            if np.max(img_num_per_cls) <= img_max:
                break
            elif i == (max_iter - 1):
                raise Exception('Dirichlet shift with %i iterations still failed to generate plausible subset, '
                                'please recheck the params.' % max_iter)
    else:
        probs = np.random.dirichlet([alpha] * cls_num)
        img_num_per_cls = [math.trunc(x * sample_num) for x in probs]

    if 0 in img_num_per_cls:
        logging.info('[warning] Some test class have 0 sample number after Dirichlet Shift.')

    return img_num_per_cls


def group_int_list(data, idx):
    # the idx should be output of np.unique(data), which is sorted.
    # assert len(set(idx)) == len(idx)
    idx = list(map(int, list(idx)))
    data = list(map(int, data))
    group = [[] for _ in idx]
    for i, v in enumerate(data):
        group[v].append(i)

    return group
