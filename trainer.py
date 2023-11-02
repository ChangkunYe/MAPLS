import torch
import torch.nn as nn
import numpy as np
from model.model_init import model_init, optimizer_init, lr_scheduler_init, loss_init
from model.model_train import model_train
from model.model_eval import model_eval
from model.metrics_eval import metrics_cal, logging_metrics, avg_metrics
from model.reweight import get_cls_weight
import logging


def trainer(train_set, val_set, cfg, resume=False, log_stamp='None'):
    # Get config detail for each part of model----------------------------------#
    state_dicts = cfg.state_dict

    # Load existing model if resume training------------------------------------#
    if resume:
        model_state_dict = state_dicts['model']
        optimizer_state_dict = state_dicts['optimizer']
        if 'lr_scheduler' in cfg.keys:
            lr_scheduler_state_dict = state_dicts['lr_scheduler']
        logging.info('Resume training from --modelpath: ')
        start_epoch = cfg.train_info['current_epoch'] - 1
    else:
        start_epoch = 0
        if 'model' in state_dicts.keys():
            logging.info('Not resume but find \'model\' in \'state_dict\', fine-tune the model.')
            model_state_dict = state_dicts['model']
        else:
            logging.info('Training from scratch.')
            model_state_dict = None
        optimizer_state_dict = None
        lr_scheduler_state_dict = None
    # Model Init----------------------------------------------------------------#
    # Load model to be trained
    model = model_init(cfg.model, cfg.dataset['name'], model_state_dict)
    # Load optimizer
    optimizer = optimizer_init(cfg.optimizer, model, optimizer_state_dict)
    # Load lr_scheduler
    lr_scheduler = lr_scheduler_init(cfg.lr_scheduler, optimizer, lr_scheduler_state_dict)

    logging.info('Model initialization finished.')

    # Move everything to cuda, setup data parallel------------------------------#
    gpu = cfg.model['gpu']
    if torch.cuda.device_count() == 1:
        model.cuda() if gpu is None else model.cuda(gpu)
        logging.info('Training on single GPU')
    else:
        model = nn.DataParallel(model)
        model.cuda()
        logging.info('Training parallel on %i GPUs given by CUDA_VISIBLE_DEVICES' % torch.cuda.device_count())

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    # Epoch Training start----------------------------------------------------#
    best_acc = 0. if start_epoch == 0 else cfg.train_info['best_acc']
    class_num_list = cfg.train_info['class_num_list']

    # Scale up the logit when using networks.NormLayer.NormedLinear instead of nn.Linear for last FC layer in resnet
    # (LDAM loss use 30.)
    logit_scale = cfg.loss['scale']
    logging.info('Scale the output logit by %.2f' % logit_scale)

    # Label Shift parameter initialize
    train_probs = []
    train_labels = []
    val_probs = []

    for epoch in range(cfg.train_info['epoch'])[start_epoch:]:
        # Loss Reweighting setup ---------------------------------------------#
        per_cls_weights = get_cls_weight(cfg.train_info,
                                         epoch,
                                         cfg.lr_scheduler['milestones'],
                                         class_num_list,
                                         train_probs=train_probs,
                                         train_labels=train_labels,
                                         val_probs=val_probs)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda() if per_cls_weights is not None else None

        # Loss Init-----------------------------------------------------------#
        criterion = loss_init(cfg.loss,
                              cfg.train_info['class_num_list'],
                              per_cls_weights,
                              logit_scale)

        # Train set Train-----------------------------------------------------#
        loss = model_train(model,
                           criterion,
                           optimizer,
                           train_set,
                           aug_config=cfg.train_info['data_aug'])

        lr_scheduler.step()

        # Val set Eval, Checkpoint save---------------------------------------#
        if (epoch + 1) % cfg.train_info['print_log'] == 0 or (epoch + 1) == cfg.train_info['epoch']:
            logging.info('[%i/%i] epoch, training loss: %.4f' % (epoch + 1, cfg.train_info['epoch'], loss))
            # Model Eval------------------------------------------------------#
            train_probs, train_labels = model_eval(model, train_set)

            if type(val_set) != list:
                val_set = [val_set]
            all_metrics = []
            all_gt_metrics = []
            for x in val_set:
                # model evaluation
                probs, labels = model_eval(model, x)

                acc, metrics, gt_metrics = metrics_cal(probs, labels, class_num_list,
                                                       train_probs=train_probs,
                                                       train_labels=train_labels,
                                                       val_probs=probs,
                                                       val_labels=labels)
                all_metrics.append(metrics)
                all_gt_metrics.append(gt_metrics)
                logging.info('\n\n Test Set Totoal Sample Num is %i \n\n' % len(labels))
                logging.info('\n\n#-----------------Evaluation End for This Iteration-----------------#\n\n')

            for mode in ['min', 'max', 'mean']:
                metrics_mean = avg_metrics(all_metrics, mode=mode)
                model_stamp = 'test-' + '-'.join(log_stamp.split('-')[:-2])
                logging_metrics(metrics_mean, model_stamp + '-' + mode)

            [x.update(y) for x, y in zip(all_metrics, all_gt_metrics)]
            save_metric = {str(x): y for x, y in enumerate(all_metrics)}
            torch.save(save_metric, './log/metrics-' + log_stamp + '.pth')

            # Save training status--------------------------------------------#
            if acc > best_acc:
                best_acc = acc
                cfg.update(['train_info', 'best_acc'], best_acc)
                cfg.update(['train_info', 'best_epoch'], epoch + 1)
                cfg.update(['train_info', 'best_metrics'], metrics)
            logging.info('Current best Top1 Acc %.4f, at epoch [%i/%i]' %
                         (best_acc, epoch + 1, cfg.train_info['epoch']))
            cfg.update(['train_info', 'current_epoch'], epoch + 1)

            # --------------------------Save state_dicts------------------------#
            if not isinstance(model, nn.DataParallel):
                cfg.update(['state_dict', 'model'], model.state_dict())
            else:
                cfg.update(['state_dict', 'model'], model.module.state_dict())
            cfg.update(['state_dict', 'optimizer'], optimizer.state_dict())
            if lr_scheduler is not None:
                cfg.update(['state_dict', 'lr_scheduler'], lr_scheduler.state_dict())

            # -------------------------Write to checkpoint----------------------#
            cfg.save(cfg.checkpoint['save_path'])
            logging.info('Checkpoint saved at ' + cfg.checkpoint['save_path'])

    logging.info('Training Finished.')
