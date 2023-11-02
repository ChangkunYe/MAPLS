# Common Utils in Target Label Shift Estimation
import numpy as np
from typing import List


# Post hoc Label Shift Correction--------------------------------------#
def lsc(probs: np.ndarray, w: List):
    r"""
    Implementation of Label Shift Compensation (LSC) with known target label distribution.
    Given source domain P(Y_s=i) and P(Y_s=i|X_s=x), target domain P(Y_t=i),
    estimate target predicted probability q(y|x) on test set.

    Args:
        probs:      Softmax probability P(Y_s=i|X_s=x) predicted by the classifier,
                    for all samples in validation set (N x C).
        w:          Ratio of Target over Source domain label distribution $ w = P(Y_t=i) / P(Y_s=i) $,
                    Not necessarily normalized to 1.

    Shapes:
        * Input:
            probs:     N x C   (No. of samples) x (No. of classes),
            w:         C       (No. of classes),
        * Output:
            pc_probs:  N x C   (No. of samples) x (No. of classes)


    For more information see original paper:
    [2002] "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    """
    assert len(w) == probs.shape[-1]
    pc_probs = normalized(probs * w, axis=-1, order=1)

    return pc_probs


# Estimation of Source Label Distribution P(Y_s=i) or p(\hat{y}=c_i)----------#
def get_py(probs: np.ndarray, cls_num_list: List[int] = None, mode='soft'):
    r"""
    Estimation of source label distribution (normalized)
    Given source domain P(Y_s=i|X_s=x)=f(x) and No. of sample per-class,
    estimate P(Y_s=i) or p(\hat{y}=c_i)

    Args:
        probs:          Softmax probability p(\hat{y}|x) predicted by classifier,
                        over all samples on train set (N x C).
        cls_num_list:   No. of samples in each class (C)
        mode:           Method used to estimate p(\hat{y}=c_i),
                        'soft' will estimate p(\hat{y}=c_i) \approx \sum^N_j p(\hat{y}=c_i|x_j),
                        'hard' will estimate p(\hat{y}=c_i) \approx \sum^N_j \mathds{1}(\arg\max_c p(y=c|x_j)=c_i)
                        'gt' will estimate p(\hat{y}=c_i) \approx P(Y_s=i), which is given by cls_num_list

    Shapes:
        * Input:
            probs:          N x C   (No. of samples) x (No. of classes),
            cls_num_list:   C       (No. of classes),
        * Output:
            py:             C       (No. of classes)

    Examples:

        >>> import numpy as np
        >>> from numpy.linalg import norm
        >>> class_num = 5; val_set_sample_num = 10
        >>> prob = norm(np.random.normal(size=(val_set_sample_num, class_num)), ord=1, axis=-1)
        >>> num_list = list(range(class_num))
        >>> py = get_py(prob, num_list)
    """
    cls_num = probs.shape[-1]

    if mode == "soft":
        py = np.mean(probs, axis=tuple(range(len(np.shape(probs)) - 1)))
    elif mode == "hard":
        py = np.bincount(np.argmax(probs, axis=-1), minlength=cls_num)
        py = py / py.sum()
    elif mode == 'gt' and cls_num_list is not None:
        py = np.array(cls_num_list) / cls_num
    else:
        raise ValueError("'mode' only support options: 'soft', 'hard', 'gt'")

    return py / py.sum()



def get_marginal(probs: np.ndarray, cls_num: int, mode: str = 'soft'):
    r"""
    Get Marginal Distribution $P(Y=.)$ given $P(Y=.|X=x)$ by summing over x
    """
    assert (mode in ['soft', 'hard']) and probs.shape[-1] == cls_num
    if mode == 'hard':
        qz = np.zeros(cls_num)
        for i in np.argmax(probs, axis=-1):
            qz[i] += 1.
        qz = qz / qz.sum()
    elif mode == 'soft':
        qz = np.mean(probs, axis=0)

    return qz


def get_confusion_matrix(probs: np.ndarray,
                         labels: List,
                         cls_num: int,
                         mode: str = 'soft'):
    r"""
    Get Confusion Matrix of prediction given prediction $P(Y=.|X=x)$ and ground truth label $Y=.$
    """
    assert (mode in ['soft', 'hard']) and probs.shape[-1] == cls_num
    cm = np.zeros((cls_num, cls_num))
    if mode == 'soft':
        for i, j in zip(labels, probs):
            cm[i, :] += j
    elif mode == 'hard':
        labels_pred = np.argmax(probs, axis=-1)
        for i, j in zip(labels, labels_pred):
            cm[i, j] += 1

    return cm


def normalized(a, axis=-1, order=2):
    r"""
    Prediction Normalization
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def Topk_qy(probs: np.ndarray, cls_num, topk_ratio=0.8, head=0, normalize=True):
    r"""
    Get Marginal Distribution $P(Y=.)$ given Topk of $P(Y=.|X=x)$ by summing over x
    """
    assert probs.shape[-1] == cls_num

    k = np.clip(int(cls_num * topk_ratio) + head, head + 1, cls_num)
    qy = np.zeros(cls_num)
    for x in probs:
        idx = np.argsort(x)[::-1]
        idx = idx[head:k]
        qy[idx] += x[idx]

    if normalize:
        qy = qy / probs.shape[0]

    return qy
