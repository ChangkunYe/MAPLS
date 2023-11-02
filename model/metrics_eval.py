import logging
from model.metrics import *
from label_shift.label_shift_est import get_label_shift_ratio, ls_metrics_eval
from label_shift.common import lsc, get_py
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix


def metrics_cal(probs, labels, train_cls_num_list, train_probs, train_labels, val_probs, val_labels, ensemble_num=None):
    if ensemble_num is None:
        def f(x):
            return x
    else:
        def f(x):
            return x.mean(-2)

    def get_mse(a, b):
        return np.mean(np.power(a - b, 2))

    cls_num = len(train_cls_num_list)
    probs = np.array(probs)
    train_probs = np.array(train_probs)
    val_probs = np.array(val_probs)

    # Ground Truth Label Distribution and Ratios
    qy_gt = np.zeros(cls_num)
    for i in labels:
        qy_gt[i] += 1

    qy_gt = qy_gt / qy_gt.sum()
    py_all = {'gt': get_py(f(train_probs), train_cls_num_list, mode="gt"),
              'soft': get_py(f(train_probs), train_cls_num_list, mode="soft"),
              'hard': get_py(f(train_probs), train_cls_num_list, mode="hard")}
    w_gt = qy_gt / py_all['gt']
    gt_metrics = {'ground_truth': {'py': py_all, 'qy': qy_gt, 'w': w_gt}}
    metrics = {}

    # Regular Accuracy--------------------------------------------------------#
    logging.info('#-----Normal Softmax Metric------#')
    normal_probs = probs.copy()
    normal_metric = get_metrics(f(normal_probs), labels, train_cls_num_list)
    del normal_probs
    metrics['nomral'] = normal_metric

    # Label Shift Compensation Accuracy---------------------------------------#
    logging.info('#-----PC Softmax Metric------#')
    for mode in ['gt', 'soft']:
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(val_probs), train_cls_num_list,
                                      py=py_all[mode], qy_method='uniform')
        pc_probs = f(lsc(probs, w))

        logging.info('#Mode: %s' % mode)
        pc_metric = get_metrics(pc_probs, labels, train_cls_num_list)
        del pc_probs

        w_mse = get_mse(w, w_gt)
        logging.info('w MSE: %.8f' % w_mse)

        pc_metric['w_mse'] = w_mse
        pc_metric['w'] = w
        pc_metric['qy'] = qy

        metrics['pc_' + mode] = pc_metric

    # Known Target Distribution Label Shift Accuracy--------------------------#
    logging.info('#-----Known Q(y) PC Softmax Metric------#')
    val_cls_num_list = np.zeros(cls_num)
    for i in val_labels:
        val_cls_num_list[i] += 1

    for mode in ['gt', 'soft']:
        w, qy = get_label_shift_ratio(f(train_probs), train_labels, f(val_probs),
                                      train_cls_num_list, py_all[mode],
                                      val_cls_num_list=val_cls_num_list,
                                      qy_mode=mode, qy_method='known')
        qy_probs = f(lsc(probs, w))

        logging.info('#Mode: %s' % mode)
        qy_metric = get_metrics(qy_probs, labels, train_cls_num_list)
        del qy_probs

        w_mse = get_mse(w, w_gt)
        logging.info('w MSE: %.8f' % w_mse)

        qy_metric['w_mse'] = w_mse
        qy_metric['w'] = w
        qy_metric['qy'] = qy

        metrics['qy_' + mode] = qy_metric

    # Unknown Target Distribution Label Shift Accuracy------------------------#
    metrics.update(ls_metrics_eval(probs, labels, train_probs, train_labels, val_probs, val_labels,
                                   train_cls_num_list, get_metrics, ensemble_num, py_all=py_all))

    return normal_metric['acc'], metrics, gt_metrics


def get_metrics(probs, labels, cls_num_list):
    pred = np.argmax(probs, axis=-1)
    acc = acc_cal(probs, labels, method='top1')
    logging.info('Evaluation Top1 Acc %.4f' % acc)

    matrix = confusion_matrix(labels, pred)
    per_cls_acc = matrix.diagonal() / matrix.sum(axis=1)
    avg_per_cls_acc = per_cls_acc.mean() * 100
    logging.info('Per-Class Top1 Acc %.4f' % (avg_per_cls_acc))

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))

    # precision, recall, f1, _ = prfs(labels, pred, average='micro')
    # logging.info('(micro) Precision: %.4f, Recall: %.4f, F Score: %.4f' % (precision, recall, f1))

    precision, recall, f1, support = prfs(labels, pred, average='macro', zero_division=0)
    logging.info('(macro) Precision: %.4f, Recall: %.4f, F Score: %.4f' % (precision, recall, f1))

    # pc_ece = ece_loss(torch.Tensor(np.array(pc_probs)), torch.LongTensor(np.array(labels))).detach().cpu().numpy()
    ece = ECECal(np.array(probs), list(labels))
    sce = SCECal(np.array(probs), list(labels), len(cls_num_list))
    bier = BierCal(np.array(probs), list(labels))
    ent = EntropyCal(np.array(probs))
    logging.info('ECE, SCE, Bier, Entropy of current model: %.4f, %.4f, %.4f, %.4f' % (ece, sce, bier, ent))

    result = {'acc': acc,
              'sce': sce,
              'ece': ece,
              'bier': bier,
              'entropy': ent,
              'mmf_acc': mmf_acc,
              'cls_acc': avg_per_cls_acc,
              'precision': precision,
              'recall': recall,
              'f1': f1}

    result = round_val(result)

    return result


def round_val(metrics):
    # Round metric values to a more reader friendly form
    for k, v in metrics.items():
        if type(v) in [np.ndarray, list]:
            metrics[k] = [round(float(x), 4) for x in list(v)]
        else:
            metrics[k] = round(float(v), 4)

    return metrics


def logging_metrics(metrics, stamp='default'):
    key_names = metrics.keys()

    logging.info('#--------------Metric Table Start-------------------#')
    logging.info('Model Name: ' + ', '.join(key_names))
    acc = ['{:.2f}'.format(float(v['acc'])) for k, v in metrics.items()]
    w_mse = ['{:.4f}'.format(float(v['w_mse'])) if 'w_mse' in v.keys() else 'None' for k, v in metrics.items()]
    logging.info('Top1 Acc: ' + ' & '.join(acc))
    logging.info('w_mse: ' + ' & '.join(w_mse))

    content = acc + w_mse
    label = list(metrics.keys())
    labels = label + label
    try:
        with open('./log/table-' + stamp + '.txt', 'r') as f:
            lines = f.read().splitlines()
        with open('./log/table-' + stamp + '.txt', 'w') as f:
            for line, x in zip(lines, content):
                f.write(line + ' & ' + x + '\n')
    except FileNotFoundError:
        with open('./log/table-' + stamp + '.txt', 'w') as f:
            for x, y in zip(labels, content):
                f.write(x + " & " + y + '\n')

    logging.info('#--------------Metric Table End---------------------#')


def avg_metrics(metrics_list, mode='mean'):
    avg_metrics = {}
    # metrics_num = len(metrics_list)
    if mode =='mean':
        fun = np.mean
    elif mode == 'max':
        fun = np.max
    elif mode == 'min':
        fun = np.min
    else:
        raise Exception('Mode for evaluating metrics list should in [\'min\',\'max\',\'mean\']')

    for k1, v1 in metrics_list[0].items():
        avg_metrics[k1] = {}
        for k2, v2 in metrics_list[0][k1].items():
            avg_metrics[k1][k2] = fun([x[k1][k2] for x in metrics_list], axis=0)

    return avg_metrics
