import torch
import torch.nn as nn
import logging
from model.model_init import model_init
from model.model_eval import model_eval
from model.metrics_eval import metrics_cal, logging_metrics, avg_metrics


def tester(dataset, trainset, valset, config, log_stamp):
    # --------------------------Load Model------------------------------------#
    model_state_dict = config['state_dict']['model']

    model = model_init(config['model'],
                       config['dataset']['name'],
                       model_state_dict)

    # ------------Move everything to cuda, setup data parallel----------------#
    if config['model']['gpu'] is not None:
        model.cuda(config['model']['gpu'])
        logging.info('Training on single GPU: %i' % config['model']['gpu'])
    else:
        model = nn.DataParallel(model)
        model.cuda()
    # ------------------------Test_set Eval-----------------------------------#
    class_num_list = config['train_info']['class_num_list']

    # p(\hat{y}|x) evaluation for Label Shift estimation ---------------------#
    train_probs, train_labels = model_eval(model, trainset, ls_est=True)

    if type(dataset) != list:
        dataset = [dataset]
        valset = [valset]
    all_metrics = []
    all_gt_metrics = []
    for x, y in zip(dataset, valset):
        val_probs, val_labels = model_eval(model, y, ls_est=False)

        # model evaluation
        probs, labels = model_eval(model, x)

        acc, metrics, gt_metrics = metrics_cal(probs, labels, class_num_list,
                                               train_probs=train_probs,
                                               train_labels=train_labels,
                                               val_probs=val_probs,
                                               val_labels=val_labels)
        all_metrics.append(metrics)
        all_gt_metrics.append(gt_metrics)
        logging.info('\n\n Test Set Totoal Sample Num is %i \n\n' % len(labels))
        logging.info('\n\n#-----------------Evaluation End for This Iteration-----------------#\n\n')

    for mode in ['min', 'max', 'mean']:
        metrics_mean = avg_metrics(all_metrics, mode=mode)
        table_stamp = log_stamp.split('-')[:-2]
        model_stamp = 'test-' + '-'.join(table_stamp)
        logging_metrics(metrics_mean, model_stamp + '-' + mode)

    [x.update(y) for x, y in zip(all_metrics, all_gt_metrics)]
    save_metric = {str(x): y for x, y in enumerate(all_metrics)}
    torch.save(save_metric, './log/metrics-' + log_stamp + '.pth')

    logging.info("Test program finished.")
    return acc
