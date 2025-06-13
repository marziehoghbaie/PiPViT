import inspect
import os
import sys
import time

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# local imports

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)

from utils.utils import *
from test_files.test import test
from utils.loss import calculate_loss
from utils.koleo_loss import KoLeoLoss
from sklearn.metrics import f1_score


def pretrain(train_config,
             logger,
             model,
             optimizer,
             pretrain_loader,
             scheduler,
             ckpt_saver,
             init_epoch=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() and train_config['use_gpu'] else 'cpu')
    logger.info('model will be running on {}'.format(device))

    model.train()
    # train loop
    scaler = torch.cuda.amp.GradScaler()
    model.cuda()
    iters = len(pretrain_loader)
    model.train()
    best_loss = 1000.0
    koleo_loss = KoLeoLoss()
    best_checkpoint_path = None
    for epoch in range(init_epoch, train_config['num_epoch_pretrain']):
        model.train()
        # loop over dataset
        accumulated_loss = []
        align_pf_weight = (epoch / (train_config['num_epoch_pretrain'] - train_config['init_epoch_pretrain'])) * 1.

        logger.info('[INFO] Current learning rate at epoch {}: {}'.format(epoch, showLR(optimizer)))
        print('[INFO] Pre-Train loop ...')
        epoch_time = time.time()
        for idx, (images_1, images_2, _) in (enumerate(pretrain_loader)):
            optimizer.zero_grad(set_to_none=True)
            # Forward Pass
            images_1, images_2 = map(Variable, (images_1, images_2))
            images_1, images_2 = images_1.to(device), images_2.to(device)
            _, proto_features, pooled = model(torch.cat([images_1, images_2]))
            loss = calculate_loss(proto_features,
                                  pooled,
                                  align_pf_weight,
                                  train_config['t_weight'],
                                  train_config['unif_weight'],
                                  tanh_type=train_config['tanh_type'],
                                  EPS=1e-10)

            # koleo loss
            koleo_loss_weight = train_config['koleo_loss_weight']
            proto_features = proto_features.reshape(proto_features.shape[0]*proto_features.shape[1], proto_features.shape[2]*proto_features.shape[3]) # initial shape: torch.Size([16, 768, 12, 12]), after reshape torch.Size([16, 768, 144])
            loss += koleo_loss_weight * sum(koleo_loss(p) for p in proto_features.chunk(2))
            # end koleo loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + idx / iters)
            accumulated_loss.append(loss.item())
        accumulated_loss = np.average(accumulated_loss)
        logger.info(
            f'[INFO] Pretrain epoch {epoch}th took {time.time() - epoch_time} secs with average accumulated loss: {accumulated_loss}')
        if accumulated_loss < 0.:
            logger.info('[INFO] The loss is negative. The pretraining is stopped.')
            break

        if accumulated_loss < best_loss:
            save_dict = {
                'epoch_idx': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            best_checkpoint_path = ckpt_saver.save(save_dict, np.array(np.average(accumulated_loss)))
            best_loss = accumulated_loss
        elif accumulated_loss < 0:
            logger.info('[INFO] The loss is not decreased. The pretraining is stopped.')
            break

    return best_checkpoint_path


def train_val_all_losses(train_config,
                         logger,
                         model,
                         optimizer,
                         loss_fn,
                         train_loader,
                         val_loader,
                         scheduler,
                         ckpt_saver,
                         init_epoch=0):
    writer = SummaryWriter(log_dir=ckpt_saver.save_dir)
    best_checkpoint_path = None
    best_acc = 0.0
    device = torch.device('cuda:0' if torch.cuda.is_available() and train_config['use_gpu'] else 'cpu')
    logger.info('model will be running on {}'.format(device))
    epochs_acc = []
    epochs_loss = []
    epochs_loss_val = []
    epochs_acc_val = []

    model.train()
    # train loop
    scaler = torch.cuda.amp.GradScaler()
    num_updates = 0  # applicable for InverseSquareRootScheduler
    model.cuda()
    iters = len(train_loader)
    logger.info('[INFO] Training with Tanh + {} loss'.format(train_config['tanh_type']))
    koleo_loss = KoLeoLoss()
    for epoch in range(init_epoch, train_config['num_epochs']):
        model.train()
        # loop over dataset
        logger.info('[INFO] Current learning rate at epoch {}: {}'.format(epoch, showLR(optimizer)))
        logger.info('[INFO] Train loop ...')
        running_corrects = 0
        running_loss = 0
        running_all = 0
        epoch_time = time.time()
        for idx, (images_1, images_2, labels) in (enumerate(train_loader)):
            labels = torch.cat([labels, labels])
            optimizer.zero_grad(set_to_none=True)
            # Forward Pass
            images_1, images_2 = map(Variable, (images_1, images_2))
            images_1, images_2 = images_1.to(device), images_2.to(device)
            logits, proto_features, pooled = model(torch.cat([images_1, images_2]), inference=False)

            if train_config['loss']=='nll':
                softmax_inputs = torch.log1p(logits ** model._classification.normalization_multiplier)
                loss_cre = loss_fn(F.log_softmax((softmax_inputs), dim=1), labels.cuda())
            else:
                loss_cre = loss_fn(logits, labels.cuda())

            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            if train_config['loss'] == 'bce':
                output = torch.round(torch.sigmoid(logits))
                running_corrects += f1_score(labels.squeeze().detach().cpu().numpy(),
                                             output.squeeze().detach().cpu().numpy(),
                                             average='macro') * (images_1.size(0) + images_2.size(0))
            else:
                running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            current_lr = showLR(optimizer)

            # Compute Loss and Perform Back-propagation
            loss = (calculate_loss(proto_features,
                                   pooled,
                                   train_config['align_pf_weight_train'],
                                   train_config['t_weight_train'],
                                   train_config['unif_weight_train'],
                                   EPS=1e-10,
                                   tanh_type=train_config['tanh_type'])
                    + train_config['cross_entrp_coef'] * loss_cre)

            # koleo loss
            koleo_loss_weight = train_config['koleo_loss_weight_train']
            if koleo_loss_weight > 0:
                proto_features = proto_features.reshape(proto_features.shape[0]*proto_features.shape[1], proto_features.shape[2]*proto_features.shape[3]) # initial shape: torch.Size([16, 768, 12, 12]), after reshape torch.Size([16*768, 144])
                loss += koleo_loss_weight * sum(koleo_loss(p) for p in proto_features.chunk(2))
            # end koleo loss

            running_loss += loss.item() * (images_1.size(0) + images_2.size(0))
            running_all += images_1.size(0)
            running_all += images_2.size(0)

            # Scale Gradients
            scaler.scale(loss).backward()
            # Update Optimizer
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + idx / iters)

            # tensorboard setting
            writer.add_scalar('ACC/train/batch', running_corrects / running_all, idx)
            writer.add_scalar('Loss/train/batch', running_loss / running_all, idx)

            if 0 == idx % 50:
                logger.info("batch id: {}, batch acc: {}, batch loss: {}".format(idx,
                                                                                 running_corrects / running_all,
                                                                                 running_loss / running_all))
                logger.info('[INFO] Current learning rate: {}'.format(showLR(optimizer)))
            num_updates += 1

            """clipping the weights smaller than 1e-3"""
            if train_config['clamp'] and (
                    epoch == train_config['num_epochs'] or epoch % train_config['clamp_frequency'] == 0) and \
                    train_config[
                        'num_epochs'] > 1 and 0 < epoch < train_config['stop_clamp_epoch']:
                with torch.no_grad():
                    if idx == 1:
                        logger.info('weights are clamped...')
                    model._classification.weight.copy_(torch.clamp(model._classification.weight.data - train_config['clamp_value'],
                                                        min=0.))  # set weights in classification layer < 1e-3 to zero
                    model._classification.normalization_multiplier.copy_(
                        torch.clamp(model._classification.normalization_multiplier.data, min=1.0))
                    if model._classification.bias is not None:
                        model._classification.bias.copy_(torch.clamp(model._classification.bias.data, min=0.))
            """end weight clipping"""

        print('[INFO] Validation loop ...')
        logger.info('[INFO] This training epoch took {} sec'.format(time.time() - epoch_time))
        val_time = time.time()
        val_acc, val_loss = test(val_loader, model, loss_fn, logger, phase='val', train_config=train_config)
        logger.info('[INFO] This validation epoch took {} sec'.format(time.time() - val_time))

        # -- save the best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            if best_acc > 0.5:
                save_dict = {
                    'epoch_idx': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                best_checkpoint_path = ckpt_saver.save(save_dict, val_acc)
                logger.info('[INFO] the checkpoint is saved at epoch {} and with val acc {}'.format(epoch, val_acc))

        epoch_acc = running_corrects / running_all
        epoch_loss = running_loss / running_all

        # Tensorboard setting
        # train
        writer.add_scalar('ACC/train', epoch_acc, epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # val
        writer.add_scalar('ACC/val', val_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info("epoch: {}, epoch acc: {}, epoch loss: {},"
                    " val acc: {}, val loss: {} with current lr {}".format(epoch,
                                                                           epoch_acc,
                                                                           epoch_loss,
                                                                           val_acc,
                                                                           val_loss,
                                                                           current_lr))
        # # # # # # # # # # # #
        epochs_acc.append(epoch_acc)
        epochs_loss.append(epoch_loss)

        epochs_acc_val.append(val_acc)
        epochs_loss_val.append(val_loss)

    writer.close()
    draw_results([epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val], save_path=ckpt_saver.save_dir)
    return model, best_checkpoint_path
