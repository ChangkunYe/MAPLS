import os
import pickle
import torchvision.datasets as dset
from torch.utils.data import DataLoader as DLoader
from torch.utils.data import Subset
import numpy as np
from data_loader.sampler import get_sampler
from data_loader.imb_dataset_gen import create_imb_subset
from torchvision import transforms
from PIL import Image
import logging

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
}


def load_data(datapath, data_transforms, params, imb_test_config=None):
    # kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory'], 'drop_last': True}
    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory']}
    if params['name'] == 'CIFAR10':
        train_dataset = CIFAR10p(root=datapath, train=True, download=False, transform=data_transforms['train'])
        val_dataset = CIFAR10p(root=datapath, train=False, download=False, transform=data_transforms['test'])
        test_dataset = CIFAR10p(root=datapath, train=False, transform=data_transforms['test'])
    elif params['name'] == 'CIFAR100':
        train_dataset = CIFAR100p(root=datapath, train=True, download=False, transform=data_transforms['train'])
        val_dataset = CIFAR100p(root=datapath, train=False, download=False, transform=data_transforms['test'])
        test_dataset = CIFAR100p(root=datapath, train=False, transform=data_transforms['test'])
    else:
        raise Exception('Dataset must in CIFAR10/100')

    if params['imb_factor'] is not None:
        train_imb_cfg = {"mode": "LT",
                         "order": "normal",
                         "imb_factor": params['imb_factor']}
        train_loc_list, train_cls_num_list = create_imb_subset(train_dataset.targets,
                                                               None,
                                                               train_imb_cfg,
                                                               return_count=True)
        train_dataset = Subset(train_dataset, train_loc_list)
    else:
        _, train_cls_num_list = np.unique(train_dataset.targets, return_counts=True)
        train_cls_num_list = train_cls_num_list.tolist()


    logging.info('Train set class num list is: ' + str(train_cls_num_list))
    dset_info = {'class_num': len(train_cls_num_list),
                 'per_class_img_num': train_cls_num_list}

    # --------------------------Define Sample Strategy--------------------------#
    sampler = get_sampler(params['sampler'], train_dataset, dset_info['per_class_img_num'])
    # ------------------Imbalance Test set if required--------------------------#
    if imb_test_config is not None:
        if 'testset_num' in imb_test_config.keys():
            test_dataset = [Subset(test_dataset,
                                   create_imb_subset(test_dataset.targets,
                                                     np.argsort(dset_info['per_class_img_num']),
                                                     imb_test_config
                                                     )
                                   )
                            for _ in range(imb_test_config['testset_num'])]
            val_dataset = [Subset(val_dataset,
                                  create_imb_subset(val_dataset.targets,
                                                    np.argsort(dset_info['per_class_img_num']),
                                                    imb_test_config
                                                    )
                                  )
                           for _ in range(imb_test_config['testset_num'])]
        else:
            test_dataset = Subset(test_dataset,
                                  create_imb_subset(test_dataset.targets,
                                                    np.argsort(dset_info['per_class_img_num']),
                                                    imb_test_config
                                                    )
                                  )
            val_dataset = Subset(val_dataset,
                                 create_imb_subset(val_dataset.targets,
                                                   np.argsort(dset_info['per_class_img_num']),
                                                   imb_test_config)
                                 )

    # ---------------------Create Batch Dataloader------------------------------#
    if imb_test_config is not None and 'testset_num' in imb_test_config.keys():
        train_set = DLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, **kwargs)
        val_set = [DLoader(x, batch_size=params['batch_size'], shuffle=False, **kwargs)
                   for x in val_dataset]
        test_set = [DLoader(x, batch_size=params['batch_size'], shuffle=False, **kwargs)
                    for x in test_dataset]
    else:
        train_set = DLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, **kwargs)
        val_set = DLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)
        test_set = DLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)

    return train_set, val_set, test_set, dset_info


# from https://discuss.pytorch.org/t/get-file-names-and-file-path-using-pytorch-dataloader/124920/3
class CIFAR10p(dset.CIFAR10):  # <----Important
    def __init__(self, root, train, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.filenames = []
        self.get_filenames()  # <----Important

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        filename = self.filenames[index]  # <----Important

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, filename  # <----Important

    def get_filenames(self):  # <----Important
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                self.entry = pickle.load(f, encoding='latin1')
            self.filenames.extend(self.entry["filenames"])


# from https://discuss.pytorch.org/t/get-file-names-and-file-path-using-pytorch-dataloader/124920/3
class CIFAR100p(dset.CIFAR100):  # <----Important
    def __init__(self, root, train, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.filenames = []
        self.get_filenames()  # <----Important

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        filename = self.filenames[index]  # <----Important

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, filename  # <----Important

    def get_filenames(self):  # <----Important
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                self.entry = pickle.load(f, encoding='latin1')
            self.filenames.extend(self.entry["filenames"])
