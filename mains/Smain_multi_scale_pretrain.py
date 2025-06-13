import argparse
import inspect
import os
import sys

import torch

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
    dataset_info['test_with_validation'] = False

    img_resolutions = [224, 384, 512]
    init_epoch = 0
    logger.info('[INFO] Config path: {}'.format(config_path))
    if train_config['pretrain_network']:
        from train_files.train import pretrain
        logger.info('[INFO] Pretrain phase has started ...')
        best_checkpoint_path = None
        train_config['num_epoch_pretrain'] = 10
        for idx, resolution in enumerate(img_resolutions): # train the model for 10 epoch for each resolution
            model_config['image_size'] = resolution
            current_results_path = os.path.join(results_path, 'resolution_{}'.format(resolution))
            os.makedirs(current_results_path, exist_ok=True)
            """ Create model """
            logger.info('[INFO] Model is changed for resolution {}'.format(model_config['image_size']))
            model = create_model(model_config, logger)
            if best_checkpoint_path is not None and idx > 0:
                model = load_model(best_checkpoint_path, model, allow_size_mismatch=True, device=device)
                logger.info('[INFO] Model is loaded from {}'.format(best_checkpoint_path))

            _, _, _, _, pretrain_loader = create_dataloaders(dataset_info, model_config, train_config, logger, where)

            if train_config['train']:
                len_train_loader = len(pretrain_loader)
                logger.info('[INFO] sample distribution in dataset {}:\n # train samples:{}'.format(
                    dataset_info['dataset_name'],
                    len(pretrain_loader)))
                logger.info(
                    f'Bscan dataset with {len(pretrain_loader.dataset.classes)} categories: {pretrain_loader.dataset.classes}')

            train_config['loss'] = model_config['loss']

            optimizer = create_optimizer(train_config, model)
            scheduler = create_scheduler(train_config, optimizer, len_train_loader)

            ckpt_saver = CheckpointSaver(current_results_path)

            train_config['include_all_loss_in_training'] = model_config['include_all_loss_in_training']

            logger.info('[INFO] Pretraining the model for {} epochs with image resolution {}'.format(train_config['num_epoch_pretrain'], model_config['image_size']))
            best_checkpoint_path = pretrain(train_config,
                                            logger,
                                            model,
                                            optimizer,
                                            pretrain_loader,
                                            scheduler,
                                            ckpt_saver,
                                            init_epoch=init_epoch)
            assert model is not None, 'Pretraining failed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = args.config_path
    main(config_path)
