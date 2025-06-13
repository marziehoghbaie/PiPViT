import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from utils.utils import transformers, TwoAugSupervisedDataset


def create_dataloaders(dataset_info, model_config, train_config, logger, where):
    transformation_train, transformation_pretrain_1, transformation_pretrain_2, transformation_test = transformers(
        model_config)
    train_loader_double = None
    test_loader = None
    train_loader = None
    val_loader = None
    pretrain_loader = None
    cuda = train_config['use_gpu'] and torch.cuda.is_available()
    train_set_double = None

    pretrain_set = None
    train_set = None
    val_set = None
    test_set = None

    if dataset_info['test_with_validation']:
        test_path = dataset_info['data_path'] + '/val'
    else:
        test_path = dataset_info['data_path'] + '/test'
    logger.info(f'[INFO] test will be run on {test_path}')

    # load test set

    if not train_config['train']:
        logger.info('[INFO] load test set from {}...'.format(dataset_info['data_path']))
        test_set = torchvision.datasets.ImageFolder(test_path, transform=transformation_test)
    else:
        logger.info('[INFO] load train set from {}...'.format(dataset_info['data_path']))
        train_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/train',
                                                     transform=transformation_train)
        if dataset_info['separate_pretrain']:
            pretrain_set = torchvision.datasets.ImageFolder(dataset_info['pretrain'])
            logger.info('[INFO] Separate pretrain set is loaded from {}'.format(dataset_info['pretrain']))
        else:
            pretrain_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/train')

        pretrain_set = TwoAugSupervisedDataset(pretrain_set,
                                               transform1=transformation_pretrain_1,
                                               transform2=transformation_pretrain_2)
        _train_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/train')
        train_set_double = TwoAugSupervisedDataset(_train_set,
                                                   transform1=transformation_pretrain_1,
                                                   transform2=transformation_pretrain_2)
        val_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/val',
                                                   transform=transformation_test)

    if train_set_double is not None:
        train_loader_double = DataLoader(train_set_double,
                                         batch_size=train_config['batch_size'],
                                         shuffle=dataset_info['shuffle'],
                                         num_workers=dataset_info['num_workers'], drop_last=True)
    if pretrain_set is not None:
        pretrain_loader = DataLoader(pretrain_set,
                                     batch_size=train_config['pretrain_batchsize'],
                                     shuffle=dataset_info['shuffle'],
                                     pin_memory=cuda,
                                     num_workers=dataset_info['num_workers'],
                                     worker_init_fn=np.random.seed(1),
                                     drop_last=True)
        train_loader = DataLoader(train_set,
                                  batch_size=train_config['batch_size'],
                                  shuffle=dataset_info['shuffle'],
                                  num_workers=dataset_info['num_workers'], drop_last=True)
        val_loader = DataLoader(val_set,
                                batch_size=train_config['batch_size'],
                                shuffle=False,
                                num_workers=dataset_info['num_workers'], drop_last=True)

    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=train_config['batch_size'],
                                 shuffle=False,
                                 num_workers=dataset_info['num_workers'])
    return train_loader, train_loader_double, val_loader, test_loader, pretrain_loader

def create_dataloaders_ood(dataset_info, model_config, train_config, logger, where):
    _, _, _, transformation_test = transformers(model_config)
    in_test_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/in_dist',
                                                       transform=transformation_test)
    out_test_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/out_dist',
                                                       transform=transformation_test)
    in_loader = DataLoader(in_test_set,
                           batch_size=train_config['batch_size'],
                           shuffle=False,
                           num_workers=dataset_info['num_workers'])

    out_loader = DataLoader(out_test_set,
                           batch_size=train_config['batch_size'],
                           shuffle=False,
                           num_workers=dataset_info['num_workers'])


    logger.info(f'[INFO] in_dist test will be run on {dataset_info["data_path"] + "/in_dist"}')
    logger.info(f'[INFO] out_dist test will be run on {dataset_info["data_path"] + "/out_dist"}')
    logger.info(f'[INFO] in_dist classes: {in_test_set.classes}')
    logger.info(f'[INFO] out_dist classes: {out_test_set.classes}')
    return in_loader, out_loader