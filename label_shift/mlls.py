import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from label_shift.common import normalized


# Estimation of Target Label Distribution P(Y_t=i) via MLLS-------------------#
def mlls(probs: np.ndarray,
         pz: List,
         max_iter: int = 20,
         init_mode: str = 'identical'):
    r"""
    Implementation of "Maximium Likelihood Label Shift (MLLS)" model
    for unknown target label distribution estimation.

    Given source domain P(Y_s=i|X_s=x) = f(x) and P(Y_s=i),
    estimate targe domain P(Y_t=i) on test set

    Args:
        probs:          Softmax probability P(Y_s=i|X_s=x) = f(x) predicted by the NN model,
                        for all samples in validation set (N x C)
        pz:             Source domain discrete label distribution $ P(Y_s=i) $, not necessarily normalized to 1.
                        It is recommended to use per-class sample num as pz.
        max_iter:       Maximum iterations for the model, max_iter=7 recommended by original paper.
        init_mode:      Mode to initialize target label distribution $ P(Y_t=i) $,
                        'uniform':    P(Y_t=i) = [1/N] * N
                        'identical':  P(Y_t=i) = P(Y_s=i)
        return_history: If True, return history of P(Y_t=i) recorded in every iteration.

    Shape:
        * Input:
            probs:      N x C    (No. of samples in val_set) x (No. of classes),
            pz:         C        (No. of classes)
        * Output:
            qz:         C        (No. of classes)

    Examples:

        >>> import numpy as np
        >>> from numpy.linalg import norm
        >>> class_num = 100; val_set_sample_num = 1000
        >>> prob = norm(np.random.normal(size=(val_set_sample_num, class_num)), ord=1, axis=-1)
        >>> py = [1] * class_num
        >>> qy = MLLS(prob, py)

    Reference:
        * Original paper:
        [ICML 2020] "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation"
        < http://proceedings.mlr.press/v119/alexandari20a/alexandari20a.pdf >

        * Follow-up paper:
        [NeurlPS 2020] "A Unified View of Label Shift Estimation"
        < https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf >
    """

    if type(max_iter) != int or max_iter < 0:
        raise Exception('max_iter should be a positive integer, not ' + str(max_iter))

    cls_num = len(pz)
    if init_mode == 'uniform':
        qz = np.ones(cls_num) / cls_num
    elif init_mode == 'identical':
        qz = pz.copy()
    else:
        raise ValueError('init_mode should be either "uniform" or "identical"')

    for _ in range(max_iter):
        # E-Step--------------------------------------------------------------#
        w = (np.array(qz) / np.array(pz))
        mlls_probs = normalized(probs * w, axis=-1, order=1)

        # M-Step--------------------------------------------------------------#
        qz_new = np.mean(mlls_probs, axis=tuple(range(len(np.shape(probs)) - 1)))
        # print(np.shape(pc_probs), np.shape(pred), np.shape(cls_num_list_t))

        qz = qz_new.copy()
        qz /= qz.sum()

    if np.sum(qz < 0) > 0:
        print('[warning] Negative value exist in MLLS estimation of qz, will be clip to 0')
        qz = np.clip(qz, 0, None)

    return qz



