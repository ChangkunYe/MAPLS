import numpy as np
from typing import List
from label_shift.common import get_confusion_matrix, get_marginal
import logging


# Estimation of Target Label Distribution P(Y_t=i) via BBSE-------------------#
def bbse(train_probs: np.ndarray,
         train_labels: List,
         test_probs: List,
         cls_num: int,
         py_mode: str = 'soft',
         qy_mode: str = 'soft'):
    r"""
    Implementation of Black Box Label Shift estimator (BBSE).

    Given source domain predicted p(z=i|x) = f(x), source domain ground truth P(Y_s=i).
    Given target domain predicted q(z=i|x) = f(x).
    Solve q(z) = \sum_y p(z,y) q(y)/p(y) for q(y)/p(y)
    Estimate target domain P(Y_t=i), returns w = P(Y_t=i)/P(Y_s=i)

    Args:
        train_probs:    Predicted probability of train set samples, which come from source domain.
        train_labels:   Ground truth labels of train set samples, follows source domain label distribution.
        test_probs:     Predicted probability of test set or validation set samples, which come from target domain.
        cls_num:        Total number of classes in the classification dataset.
        py_mode:        Mode for estimating source domain confusion matrix .etc, either 'soft' or 'hard'.
        qy_mode:        Mode for estimating target domain predicted label distribution, either 'soft' or 'hard'.

    Shape:
        * Input:
            train_probs:    N x C   (No. of samples in source domain set) x (No. of classes),
            train_labels:   N       (No. of samples in source domain set),
            test_probs:     M x C   (No. of samples in target domain set) x (No. of classes),
            cls_num:        1       (No. of classes C)

        * Output:
            w:              C       Estimated w = P(Y_t=i) / P(Y_s=i)

    Reference:
        * Original paper:
        [ICML 2018] Detecting and Correcting for Label Shift with Black Box Predictors
        < http://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf >

        * Official Code:
            < https://github.com/zackchase/label_shift >
        * Unofficial Code:
            < https://github.com/flaviovdf/label-shift >
    """

    assert train_probs.shape[-1] == cls_num
    assert (py_mode in ['soft', 'hard']) and (qy_mode in ['soft', 'hard'])

    pzy = get_confusion_matrix(train_probs, train_labels, cls_num, mode=py_mode).T
    pzy = pzy / len(train_labels)

    qz = get_marginal(test_probs, cls_num, mode=qy_mode)

    # lam = 1. / min(train_probs.shape[0], test_probs.shape[0])
    # w = np.linalg.solve(np.matmul(pzy.T, pzy) + lam * np.eye(cls_num), np.matmul(pzy.T, qz))

    try:
        # Solve the Ax=b if A (pzy) is not singular
        w = np.linalg.solve(np.matmul(pzy.T, pzy), np.matmul(pzy.T, qz))
        logging.info('Matrix is pseudo invertible, solved with np.linalg.solve.')
    except np.linalg.LinAlgError:
        # Go with least square solve if A (pzy) is singular
        logging.info('Matrix is singular, solved with np.linalg.lstsq.')
        w = np.linalg.lstsq(np.matmul(pzy.T, pzy), np.matmul(pzy.T, qz))
        logging.info(w.shape, w)

    if np.sum(w < 0) > 0:
        print('[warning] Negative value exist in BBSE estimation of w, will be clip to 0')
        w = np.clip(w, 0, None)

    return w


# Just for consistency check, from BBSE unofficial Code implementation: https://github.com/flaviovdf/label-shift
def calculate_marginal(y, n_classes):
    mu = np.zeros(shape=(n_classes, 1))
    for i in range(n_classes):
        mu[i] = np.sum(y == i)
    return mu / y.shape[0]


def estimate_labelshift_ratio(y_true_val, y_pred_val, y_pred_trn, n_classes):
    from sklearn.metrics import confusion_matrix
    labels = np.arange(n_classes)
    C = confusion_matrix(y_true_val, y_pred_val, labels=labels).T
    C = C / y_true_val.shape[0]

    mu_t = calculate_marginal(y_pred_trn, n_classes)
    lamb = 1.0 / min(y_pred_val.shape[0], y_pred_trn.shape[0])

    I = np.eye(n_classes)
    wt = np.linalg.solve(np.matmul(C.T, C) + lamb * I, np.matmul(C.T, mu_t))
    return wt


def estimate_target_dist(wt, y_true_val, n_classes):
    mu_t = calculate_marginal(y_true_val, n_classes)
    return wt * mu_t