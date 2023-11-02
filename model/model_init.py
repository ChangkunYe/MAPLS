import torch
import logging
import importlib

model_paths = {
    'none_cifar':    'networks.resnet_cifar',
    'none':          'networks.resnet',
    'place':         'networks.places',
}


def model_init(cfg, dataset, state_dict=None):
    if 'fc_norm' not in cfg.keys():
        cfg['fc_norm'] = False

    if dataset in ['CIFAR10', 'CIFAR100']:
        module = getattr(importlib.import_module(model_paths['none_cifar']), cfg['name'])
        model = module(cfg['output_dim'], fc_norm=cfg['fc_norm'])
    elif dataset in ['ImageNet']:
        module = getattr(importlib.import_module(model_paths['none']), cfg['name'])
        model = module(cfg['output_dim'], fc_norm=cfg['fc_norm'])
    elif dataset == 'Places':
        module = getattr(importlib.import_module(model_paths['none']), cfg['name'])
        model = module(cfg['output_dim'], fc_norm=cfg['fc_norm'])
    else:
        raise Exception('Unsupported datset name %s.' % dataset)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        logging.info('state_dict available, load model from state dict.')
    else:
        logging.info('No state_dict available, initialize model randomly.')

    return model


def optimizer_init(cfg, model, state_dict=None):
    if cfg['name'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': cfg['lr']}],
                                    lr=cfg['lr'],
                                    momentum=cfg['momentum'],
                                    nesterov=cfg['nesterov'],
                                    weight_decay=cfg['wd'])
    else:
        raise Exception('Currently only SGD optimizer is supported.')
    if state_dict is not None:
        try:
            optimizer.load_state_dict(state_dict)
        except:
            raise Exception('Failed to load state dict to current optimizer.')

    return optimizer


def lr_scheduler_init(cfg, optimizer, state_dict=None):
    if cfg['name'] == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'])
    elif cfg['name'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
    elif cfg['name'] == 'warmup_step':
        gamma = 0.1

        def lr_lambda(epoch):
            if epoch >= cfg["milestones"][1]:
                lr = gamma * gamma
            elif epoch >= cfg["milestones"][0]:
                lr = gamma
            else:
                lr = 1

            """Warmup"""
            warmup_epoch = cfg["warmup_epoch"]
            if epoch < warmup_epoch:
                lr = lr * float(1 + epoch) / warmup_epoch
            return lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise Exception('Unsupported lr_scheduler, expect in [\'cos\',\'step\',\'warmup_step\']')

    if state_dict is not None:
        lr_scheduler.load_state_dict(state_dict)

    return lr_scheduler


def loss_init(cfg, class_num_list, per_cls_weights=None, logit_scale=1):
    # --------------------------Loss Init-----------------------------------#
    if cfg['name'] == 'CE':
        from model.losses.CE_loss import CELoss
        criterion = CELoss(weight=per_cls_weights, scale=logit_scale).cuda()
    elif cfg['name'] == 'Focal':
        from model.losses.Focal_loss import FocalLoss
        criterion = FocalLoss(gamma=cfg['focal_gamma'], weight=per_cls_weights, scale=logit_scale).cuda()
    elif cfg['name'] == 'LDAM':
        from model.losses.LDAM_loss import LDAMLoss
        criterion = LDAMLoss(class_num_list, weight=per_cls_weights, s=logit_scale).cuda()
    else:
        raise Exception('Unsupported Loss type.')

    return criterion
