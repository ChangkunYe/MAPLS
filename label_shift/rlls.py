import numpy as np
import cvxpy as cp
from typing import List
from label_shift.common import get_confusion_matrix, get_marginal, get_py


def rlls(train_probs: np.ndarray,
         train_labels: List,
         test_probs: List,
         cls_num: int,
         py_mode: str = 'soft',
         qy_mode: str = 'soft',
         alpha: float = 0.01):
    r"""
    Implementation of Regularized Learning for Domain Adaptation under Label Shifts (RLLS)

    Given source domain predicted p(Y=.|X_s=x) = f(x), source domain ground truth P(Y_s=i).
    Given target domain predicted f(X_t).
    Solve q(z) = \sum_y p(z,y) q(y)/p(y) for q(y)/p(y)
    Estimate target domain P(Y_t=i), returns w = P(Y_t=i)/P(Y_s=i) = 1 + \lambda * \hat{\theta}

    Args:
        train_probs:    Predicted probability of train set samples, which come from source domain.
        train_labels:   Ground truth labels of train set samples, follows source domain label distribution.
        test_probs:     Predicted probability of test set or validation set samples, which come from target domain.
        cls_num:        Total number of classes in the classification dataset.
        py_mode:        Mode for estimating source domain confusion matrix .etc, either 'soft' or 'hard'.
        qy_mode:        Mode for estimating target domain predicted label distribution, either 'soft' or 'hard'.
        alpha:          Hyperparameter of RLLS, default 0.01 according to previous works

    Shape:
        * Input:
            train_probs:    N x C   (No. of samples in source domain set) x (No. of classes),
            train_labels:   N       (No. of samples in source domain set),
            test_probs:     M x C   (No. of samples in target domain set) x (No. of classes),
            cls_num:        1       (No. of classes C)

        * Output:
            w:              C       Estimated w = P(Y_t=i) / P(Y_s=i)


    Reference:
        * Original Paper:
        [ICLR 2019] "Regularized Learning for Domain Adaptation under Label Shifts"
        < http://tensorlab.cms.caltech.edu/users/anima/pubs/pubs/RLLS.pdf >

        * Official Code:
        < https://github.com/Angela0428/labelshift >
    """

    pzy = get_confusion_matrix(train_probs, train_labels, cls_num, mode=py_mode).T
    qz = get_marginal(test_probs, cls_num, mode=qy_mode)
    pz = get_py(probs=train_probs, mode=py_mode)

    rho = compute_3deltaC(cls_num, len(train_labels), 0.05)
    # alpha = choose_alpha(n_class, pzy, qz, py_hat, rho, true_w)
    w = compute_w_opt(pzy, qz, pz, alpha * rho)

    if np.sum(w < 0) > 0:
        print('[warning] Negative value exist in RLLS estimation of w, will be clip to 0')
        w = np.clip(w, 0, None)

    return w


# Functions belows are from official code https://github.com/Angela0428/labelshift
def compute_w_opt(C_yy, mu_y, mu_train_y, rho):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho * cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    # prob.solve()
    try:
        prob.solve(verbose=False, solver=cp.SCS)
    except cp.error.SolverError:
        prob.solve(verbose=False, solver=cp.SCS, use_indirect=True)

    w = 1 + theta.value
    # print('Estimated w is', w)

    return w


def compute_3deltaC(n_class, n_train, delta):
    rho = 3 * (2 * np.log(2 * n_class / delta) / (3 * n_train) + np.sqrt(2 * np.log(2 * n_class / delta) / n_train))
    return rho


def choose_alpha(n_class, C_yy, mu_y, mu_y_train, rho, true_w):
    alpha = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    w2 = np.zeros((len(alpha), n_class))
    for i in range(len(alpha)):
        w2[i, :] = compute_w_opt(C_yy, mu_y, mu_y_train, alpha[i] * rho)
    mse2 = np.sum(np.square(np.matlib.repmat(true_w, len(alpha), 1) - w2), 1) / n_class
    i = np.argmin(mse2)
    print("mse2, ", mse2)
    return alpha[i]
