import json
import logging
import math
import os

import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.pyplot import figure
import torchvision.transforms as transforms
from typing import Tuple, Dict
import torchvision
from torch import Tensor

transformation_raw_img = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
])



class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {

            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


def transformers(model_config):
    if model_config['channels'] == 3:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        # NORM = [[0.20253482627511976], [0.11396578943414482]]
        mean = [0.20253482627511976]
        std = [0.11396578943414482]
    normalize = transforms.Normalize(mean=mean, std=std)
    transformation_train = T.Compose([
        T.Resize((model_config['image_size'], model_config['image_size'])),
        T.RandAugment(num_ops=5,
                      magnitude=9,
                      num_magnitude_bins=31),
        transforms.Grayscale(model_config['channels']),  # convert to grayscale with three channels
        transforms.ToTensor(),
        normalize
    ])

    transformation_test = T.Compose([
        transforms.Resize(size=(model_config['image_size'], model_config['image_size'])),
        transforms.Grayscale(model_config['channels']),  # convert to grayscale with three channels
        transforms.ToTensor(),
        normalize
    ])

    transformation_pretrain_1 = transforms.Compose([
        transforms.Resize(size=(model_config['image_size'] + 32, model_config['image_size'] + 32)),
        T.RandAugment(num_ops=5,
                      magnitude=9,
                      num_magnitude_bins=31),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(model_config['image_size'] + 8, scale=(0.95, 1.))
    ])
    transformation_pretrain_2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(model_config['image_size'], model_config['image_size'])),  # includes crop
        transforms.Grayscale(model_config['channels']),  # convert to grayscale with three channels
        transforms.ToTensor(),
        normalize
    ])

    return transformation_train, transformation_pretrain_1, transformation_pretrain_2, transformation_test


def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    print('2-norm of the neural network: {:.4f}'.format(para_norm ** .5))


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile(filepath), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open(filepath) as myfile:
        content = myfile.read().splitlines()
    return content


def save_as_json(d, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def load_json(json_fp):
    assert os.path.isfile(json_fp), "Error loading JSON. File provided does not exist, cannot read: {}".format(json_fp)
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


# -- checkpoints_8_class
class CheckpointSaver:
    def __init__(self, save_dir, checkpoint_fn='ckpt.pth.tar', best_fn='ckpt.best.pth.tar',
                 best_step_fn='ckpt.best.step{}.pth.tar', save_best_step=False, lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoints files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoints per step
        """

        self.save_dir = save_dir

        # checkpoints names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoints
        self.current_best = 0

        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(
                lr_steps)
            self.best_for_stage = [0] * (len(lr_steps) + 1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoints and keeps copy if current perf is the best overall or [optional] best for current LR step
        """

        # save last checkpoints
        self.checkpoint_fn = 'current_val_acc_{}_ckpt.pth.tar'.format(current_perf)
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            self.best_fn = 'best_val_acc_{}_ckpt.pth.tar'.format(current_perf)
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(epoch)
            s_idx = sum(epoch >= l for l in self.lr_steps)
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir, self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        return checkpoint_fp

    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)


def load_model(load_path, model, optimizer_net=None, allow_size_mismatch=False,
               device="cuda:0"):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
    checkpoint = torch.load(load_path, map_location=device)
    if 'model' in checkpoint.keys():
        loaded_state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint.keys():
        loaded_state_dict = checkpoint['model_state_dict']
    else:
        loaded_state_dict = checkpoint
    if allow_size_mismatch: 
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_state_dict = model.state_dict()
        model_sizes = {k: v.shape for k, v in model_state_dict.items()}

        mismatched_params = []
        for k in loaded_sizes:
            if k not in model_state_dict or loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer_net is not None:
        optimizer_net.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer_net, checkpoint['epoch_idx'], checkpoint
    return model


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori * reduction_ratio)


class InverseSquareRootScheduler:
    def __init__(self, warmup_init_lr, warmup_end_lr, warmup_updates=4000):
        """Decay the LR based on the inverse square root of the update number.
        We also support a warmup phase where we linearly increase the learning rate
        from some initial learning rate (``--warmup-init-lr``) until the configured
        learning rate (``--lr``). Thereafter we decay proportional to the number of
        updates, with a decay factor set to align with the configured learning rate.
        During warmup::
          lrs = torch.linspace(self.lr_ori, self.warmup_end_lr, self.warmup_updates)
          lr = lrs[update_num]
        After warmup::
          decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
          lr = decay_factor / sqrt(update_num)
        """
        self.warmup_init_lr = warmup_init_lr
        """warmup the learning rate linearly for the first N updates"""
        self.warmup_updates = warmup_updates
        self.warmup_end_lr = warmup_end_lr
        # linearly warmup for the first cfg.warmup_updates
        # self.lr_step = (warmup_end_lr - self.lr_ori) / cfg.warmup_updates
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = self.warmup_end_lr * self.warmup_updates ** 0.5

        # initial learning rate
        self.lr = self.warmup_init_lr

    def adjust_lr(self, optimizer, num_updates):
        # print('[INFO] Number of current iterations is {}'.format(num_updates))
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5

        change_lr_on_optimizer(optimizer, self.lr)


def draw_results(results, save_path=None):
    assert save_path is not None

    epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val = results

    figure(figsize=(8, 6))
    plt.plot(epochs_loss)
    plt.plot(epochs_loss_val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + '/' + 'loss.png')

    figure(figsize=(8, 6))
    plt.plot(epochs_acc)
    plt.plot(epochs_acc_val)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + '/' + 'acc.png')


def create_logger(save_path, name=''):
    filename = save_path + '/' + 'log_{}.txt'.format(name)
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a+')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


def num_model_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

