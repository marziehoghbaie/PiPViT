import argparse
import inspect
import os
import sys

import torch
import torch.nn as nn


# local imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, grandgrandparentdir)
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.utils import *
from config.load_config import read_conf_file
from config.args_parser import get_args_parser
from utils.data_utils import create_dataloaders
from utils.optimizer_utils import create_optimizer
from utils.model_utils import create_model
from utils.functional import poly_loss, focal_loss
from utils.scheduler_utils import create_scheduler
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

import hashlib



def get_model_hash(model):
    model_bytes = b"".join(p.data.cpu().numpy().tobytes() for p in model.parameters())
    return hashlib.md5(model_bytes).hexdigest()


def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


def main(config_path):
    print('Torch is available ', torch.cuda.is_available())

    (model_config, dataset_info, train_config, log_info, where, config,
     device, check_point_path, model_name, results_path, logger) = read_conf_file(config_path)
    """ Create model """
    model = create_model(model_config, logger)
    logger.info('[INFO] Config path: {}'.format(config_path))
    logger.info('[INFO] model: {}'.format(model))
    """ load datasets (type of supported Datasets) """
    # check if key test_with_validation is not in the dataset_info
    if 'test_with_validation' not in dataset_info:
        dataset_info['test_with_validation'] = False # test will be run on the test set


    train_loader, train_loader_double, val_loader, test_loader, pretrain_loader = create_dataloaders(dataset_info, model_config,
                                                                                train_config, logger, where)

    if train_config['train']:
        len_train_loader = len(train_loader)
        logger.info('[INFO] sample distribution in dataset {}:\n # train samples:{}, # validation '
                    'samples {}'.format(dataset_info['dataset_name'],
                                        len(train_loader),
                                        len(val_loader)))
        logger.info(f'Bscan dataset with {len(train_loader.dataset.classes)} categories: {train_loader.dataset.classes}')
    else:
        len_train_loader = 0
        logger.info('[INFO] sample distribution in dataset {}:\n # test samples:{}'.format(dataset_info['dataset_name'],
                                        len(test_loader)))
        logger.info(f'Bscan dataset with {len(test_loader.dataset.classes)} categories: {test_loader.dataset.classes}')
    # Define classification loss function and scheduler
    if model_config['loss']=='cross_entropy':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif model_config['loss']=='nll':
        loss_fn = nn.NLLLoss(reduction='mean').to(device)
    elif model_config['loss']=='focal':
        loss_fn = focal_loss
    elif model_config['loss']=='poly':
        loss_fn = poly_loss

    train_config['loss'] = model_config['loss']

    optimizer = create_optimizer(train_config, model)
    scheduler = create_scheduler(train_config, optimizer, len_train_loader)

    ckpt_saver = CheckpointSaver(results_path)

    init_epoch = 0
    train_config['include_all_loss_in_training'] = model_config['include_all_loss_in_training']
    assert train_config['include_all_loss_in_training'], 'Only all losses are supported'
    with torch.no_grad():
        if not train_config['train']: # test mode
            logger.info('[INFO] Test Mode ...')
            if dataset_info['test_with_validation']:
                test_set_name = 'val'
            else:
                test_set_name = 'test'
            logger.info(f'[INFO] test will be run on {test_set_name}')
            model = load_model(check_point_path, model, allow_size_mismatch=False, device=device)
            logger.info('[INFO] The checkpoint is loaded from {} ...'.format(check_point_path))
            logger.info('[INFO] Model size: {:.3f} MB'.format(get_model_size(model)))
            print("sparsity ratio: ", (torch.numel(model._classification.weight) - torch.count_nonzero(
                torch.nn.functional.relu(model._classification.weight - 1e-3)).item()) / torch.numel(
                model._classification.weight),
                  flush=True)


            from test_files.test import test_complete
            logger.info('[INFO] Test phase has started ...')
            model = model.to(device)
            model.eval()
            classification_weights = model._classification.weight.cpu().detach().numpy()
            np.save(f'{results_path}/classification_weights.npy', classification_weights)
            test_complete(test_loader, model, loss_fn, logger, 'test', results_path)
            logger.info('[INFO] End Test phase ...')
            # print model size
            return
        elif train_config['resume']: # if you're loading weights in resume mode
            logger.info('[INFO] Resume Mode ...')
            model = model.to(device)
            model, optimizer, epoch_idx, ckpt_dict = load_model(check_point_path, model, optimizer,
                                                                allow_size_mismatch=train_config['allow_size_mismatch'])
            init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info('[INFO] The training is going to resume from {} ...'.format(check_point_path))
        elif train_config['pretrain']:  # if you're initializing from pretraining weights
            logger.info('[INFO] Pretrain Mode ...')
            # Hash before loading
            hash_before = get_model_hash(model)
            logger.info("Hash before loading: {}".format(hash_before))
            model = load_model(check_point_path, model, allow_size_mismatch=True, device=device)
            # Hash after loading
            hash_after = get_model_hash(model)
            logger.info("Hash after loading: {}".format(hash_after))
            logger.info('[INFO] The checkpoint is loaded from {} ...'.format(check_point_path))

    if (train_config['train'] and not train_config['resume']) or train_config['pretrain']:
        model._add_on.apply(init_weights_xavier)
        torch.nn.init.normal_(model._classification.weight, mean=1.0, std=0.1)
        args.bias = False
        if args.bias:
            torch.nn.init.constant_(model._classification.bias, val=0.)
        torch.nn.init.constant_(model._multiplier, val=train_config['power'])
        model._multiplier.requires_grad = False
        logger.info(('[INFO] multiplier initialized with value: {}'.format(model._multiplier)))
        logger.info("Classification layer initialized with mean {}".format(torch.mean(model._classification.weight).item()))
    logger.info('[INFO] training config ... \n {} \n\n'.format(config))

    if train_config['pretrain_network']: # pretrain with single resolution
        from train_files.train import pretrain
        logger.info('[INFO] Pretrain phase has started ...')
        pretrain(train_config,
             logger,
             model,
             optimizer,
             pretrain_loader,
             scheduler,
             ckpt_saver,
             init_epoch=init_epoch)
        return
    if train_config['train']:
        from train_files.train import train_val_all_losses as train_val
        train_loader = train_loader_double
        logger.info('[INFO] All losses are included in the training ...')
        logger.info('[INFO] Train phase has started ...')

        for param in model.parameters():
            param.requires_grad = True
        for param in model._classification.parameters():
            param.requires_grad = True
        for param in model._add_on.parameters():
            param.requires_grad = True
        for param in model._net.parameters():
            param.requires_grad = True
        model._multiplier.requires_grad = False

        train_val(train_config, logger, model,
                  optimizer, loss_fn, train_loader,
                  val_loader, scheduler, ckpt_saver,
                  init_epoch)
        logger.info('[INFO] End of train phase ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = args.config_path
    main(config_path)
