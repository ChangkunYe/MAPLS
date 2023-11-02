import numpy as np
from label_shift.common import normalized, Topk_qy
import logging


def mapls(train_probs: np.ndarray,
          test_probs: np.ndarray,
          pz: np.ndarray,
          qy_mode: str = 'soft',
          max_iter: int = 100,
          init_mode: str = 'identical',
          lam: float = None,
          dvg_name='kl'):
    r"""
    Implementation of Maximum A Posteriori Label Shift,
    for Unknown target label distribution estimation

    Given source domain P(Y_s=i|X_s=x) = f(x) and P(Y_s=i),
    estimate targe domain P(Y_t=i) on test set

    """
    # Sanity Check
    cls_num = len(pz)
    assert test_probs.shape[-1] == cls_num
    if type(max_iter) != int or max_iter < 0:
        raise Exception('max_iter should be a positive integer, not ' + str(max_iter))

    # Setup d(p,q) measure
    if dvg_name == 'kl':
        dvg = kl_div
    elif dvg_name == 'js':
        dvg = js_div
    else:
        raise Exception('Unsupported distribution distance measure, expect kl or js.')

    # Set Prior of Target Label Distribution
    q_prior = np.ones(cls_num) / cls_num
    # q_prior = pz.copy()

    # Lambda estimation-------------------------------------------------------#
    if lam is None:
        logging.info('Data shape: %s, %s' % (str(train_probs.shape), str(test_probs.shape)))
        logging.info('Divergence type is %s' % (dvg))
        lam = get_lamda(test_probs, pz, q_prior, dvg=dvg, max_iter=max_iter)
        logging.info('Estimated lambda value is %.4f' % lam)
    else:
        logging.info('Assigned lambda is %.4f' % lam)

    # EM Algorithm Computation
    qz = mapls_EM(test_probs, pz, lam, q_prior, cls_num,
                  init_mode=init_mode, max_iter=max_iter, qy_mode=qy_mode)

    return qz


def mapls_EM(probs, pz, lam, q_prior, cls_num, init_mode='identical', max_iter=100, qy_mode='soft'):
    # Normalize Source Label Distribution pz
    pz = np.array(pz) / np.sum(pz)
    # Initialize Target Label Distribution qz
    if init_mode == 'uniform':
        qz = np.ones(cls_num) / cls_num
    elif init_mode == 'identical':
        qz = pz.copy()
    else:
        raise ValueError('init_mode should be either "uniform" or "identical"')

    # Initialize w
    w = (np.array(qz) / np.array(pz))
    # EM algorithm with MAP estimation----------------------------------------#
    for i in range(max_iter):
        # print('w shape ', w.shape)

        # E-Step--------------------------------------------------------------#
        mapls_probs = normalized(probs * w, axis=-1, order=1)

        # M-Step--------------------------------------------------------------#
        if qy_mode == 'hard':
            pred = np.argmax(mapls_probs, axis=-1)
            qz_new = np.bincount(pred.reshape(-1), minlength=cls_num)
        elif qy_mode == 'soft':
            qz_new = np.mean(mapls_probs, axis=0)
        elif qy_mode == 'topk':
            qz_new = Topk_qy(mapls_probs, cls_num, topk_ratio=0.9, head=0)
        else:
            raise Exception('mapls mode should be either "soft" or "hard". ')
        # print(np.shape(pc_probs), np.shape(pred), np.shape(cls_num_list_t))

        # Update w with MAP estimation of Target Label Distribution qz
        # qz = (qz_new + alpha) / (N + np.sum(alpha))
        qz = lam * qz_new + (1 - lam) * q_prior
        qz /= qz.sum()
        w = qz / pz

    return qz


def get_lamda(test_probs, pz, q_prior, dvg, max_iter=50):
    K = len(pz)

    # MLLS estimation of source and target domain label distribution
    qz_pred = mapls_EM(test_probs, pz, 1, 0, K, max_iter=max_iter)

    TU_div = dvg(qz_pred, q_prior)
    TS_div = dvg(qz_pred, pz)
    SU_div = dvg(pz, q_prior)
    logging.info('weights are, TU_div %.4f, TS_div %.4f, SU_div %.4f' % (TU_div, TS_div, SU_div))

    SU_conf = 1 - lam_forward(SU_div, lam_inv(dpq=0.5, lam=0.2))
    TU_conf = lam_forward(TU_div, lam_inv(dpq=0.5, lam=SU_conf))
    TS_conf = lam_forward(TS_div, lam_inv(dpq=0.5, lam=SU_conf))
    logging.info('weights are, unviform_weight %.4f, differ_weight %.4f, regularize weight %.4f'
                 % (TU_conf, TS_conf, SU_conf))

    confs = np.array([TU_conf, 1 - TS_conf])
    w = np.array([0.9, 0.1])
    lam = np.sum(w * confs)

    logging.info('Estimated lambda is: %.4f', lam)

    return lam


def lam_inv(dpq, lam):
    return (1 / (1 - lam) - 1) / dpq


def lam_forward(dpq, gamma):
    return gamma * dpq / (1 + gamma * dpq)


def kl_div(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q + 1e-8, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def js_div(p, q):
    assert (np.abs(np.sum(p) - 1) < 1e-6) and (np.abs(np.sum(q) - 1) < 1e-6)
    m = (p + q) / 2
    return kl_div(p, m) / 2 + kl_div(q, m) / 2
