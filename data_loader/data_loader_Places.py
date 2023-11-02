# See https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/data/dataloader.py
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
from PIL import Image
import logging
from data_loader.sampler import get_sampler
from data_loader.imb_dataset_gen import create_imb_subset
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt, 'r') as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path


def load_data(datapath, data_transforms, params, imb_test_config=None):
    logging.info('Loading data from %s' % (datapath))
    dataset = params['name']
    # kwargs = {'num_workers':params['num_workers'],'pin_memory':params['pin_memory'],'drop_last':True}
    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory']}
    # ---------------------------Dataset Path-----------------------------------#
    if params['imb_factor'] is None:
        train_labelpath = datapath + '/' + dataset + '_LT_train.txt'
    elif params['imb_factor'] == 'all':
        train_labelpath = datapath + '/' + dataset + '_train.txt'
    else:
        raise Exception('Unsupported imbalance factor %s' % params['imb_factor'])
    val_labelpath = datapath + '/' + dataset + '_LT_val.txt'
    test_labelpath = datapath + '/' + dataset + '_LT_test.txt'

    # --------------------------Load Dataset from Path--------------------------#
    train_dataset = LT_Dataset(datapath, train_labelpath, data_transforms['train'])
    val_dataset = LT_Dataset(datapath, val_labelpath, data_transforms['test'])
    test_dataset = LT_Dataset(datapath, test_labelpath, data_transforms['test'])

    # -------------------------Collect Dataset Info-----------------------------#
    num_classes = len(list(set(train_dataset.targets)))
    class_loc_list = [[] for i in range(num_classes)]
    for i, label in enumerate(train_dataset.targets):
        class_loc_list[label].append(i)
    train_cls_num_list = [len(x) for x in class_loc_list]
    dset_info = {'class_num': num_classes,
                 # 'per_class_loc': class_loc_list,
                 'per_class_img_num': train_cls_num_list}
    # --------------------------Define Sample Strategy--------------------------#
    sampler = get_sampler(params['sampler'], train_dataset, train_cls_num_list)

    # ------------------Imbalance Test set if required--------------------------#
    if imb_test_config is not None:
        if 'testset_num' in imb_test_config.keys():
            test_dataset = [Subset(test_dataset,
                                   create_imb_subset(test_dataset.targets,
                                                     np.argsort(train_cls_num_list),
                                                     imb_test_config
                                                     )
                                   )
                            for _ in range(imb_test_config['testset_num'])]
            val_dataset = [Subset(val_dataset,
                                  create_imb_subset(val_dataset.targets,
                                                    np.argsort(train_cls_num_list),
                                                    imb_test_config
                                                    )
                                  )
                           for _ in range(imb_test_config['testset_num'])]
        else:
            test_dataset = Subset(test_dataset,
                                  create_imb_subset(test_dataset.targets,
                                                    np.argsort(train_cls_num_list),
                                                    imb_test_config
                                                    )
                                  )
            val_dataset = Subset(val_dataset,
                                 create_imb_subset(val_dataset.targets,
                                                   np.argsort(train_cls_num_list),
                                                   imb_test_config
                                                   )
                                 )

    # ---------------------Create Batch Dataloader------------------------------#
    train_set = DataLoader(dataset=train_dataset,
                           batch_size=params['batch_size'],
                           sampler=sampler,
                           **kwargs)

    if imb_test_config is not None and 'testset_num' in imb_test_config.keys():
        val_set = [DataLoader(dataset=x,
                              batch_size=params['batch_size'],
                              shuffle=False,
                              **kwargs)
                   for x in val_dataset]
        test_set = [DataLoader(dataset=x,
                               batch_size=params['batch_size'],
                               shuffle=False,
                               **kwargs)
                    for x in test_dataset]
    else:
        val_set = DataLoader(dataset=val_dataset,
                             batch_size=params['batch_size'],
                             shuffle=False,
                             **kwargs)
        test_set = DataLoader(dataset=test_dataset,
                              batch_size=params['batch_size'],
                              shuffle=False,
                              **kwargs)

    return train_set, val_set, test_set, dset_info
