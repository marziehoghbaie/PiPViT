import inspect
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# local imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
from utils.loss import calculate_loss
import seaborn as sns
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score


def test(loader, model, loss_fn, logger, phase, train_config):
    model.eval()
    running_corrects = 0
    running_loss = 0

    y_pred = []
    y_true = []

    if train_config['inference']:
        inference = True
    else:
        inference = False
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            logits, proto_features, pooled = model(images.cuda(), inference)
            if train_config['loss'] == 'nll':
                softmax_inputs = torch.log1p(logits ** model._classification.normalization_multiplier)
                loss_cre = loss_fn(F.log_softmax((softmax_inputs), dim=1), labels.cuda())
            else:
                loss_cre = loss_fn(logits, labels.cuda())

            probs, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            if train_config['include_all_loss_in_training']:
                loss = (calculate_loss(proto_features,
                                       pooled,
                                       train_config['align_pf_weight_train'],
                                       train_config['t_weight_train'],
                                       train_config['unif_weight_train'],
                                       EPS=1e-10,
                                       tanh_type=train_config['tanh_type'])
                        + train_config['cross_entrp_coef'] * loss_cre)
            else:
                loss = loss_cre

            running_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = running_corrects / len(loader.dataset)
    balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info('[INFO] {} acc balanced_acc, and loss: {}, {}, {}'.format(phase, accuracy, balanced_acc, loss))

    return balanced_acc, loss


def calculate_roc(y_score, y_test, logger):
    """in a multi class high imbalanced setting, micro-averaging is preferable over macro-averaging"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    if np.array(y_score).shape[1] == 2:
        micro_roc_auc_ovr = roc_auc_score(
            np.array(y_test),
            np.array(y_score)[:, 1])
        aps = average_precision_score(np.array(y_test), np.array(y_score)[:, 1], average='weighted')
    else:
        micro_roc_auc_ovr = roc_auc_score(
            y_onehot_test,
            y_score,
            multi_class="ovr",
            average="weighted")
        aps = average_precision_score(y_onehot_test, y_score, average='weighted')

    logger.info(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")
    logger.info(f"average_precision_score:\n{aps:.2f}")
    return micro_roc_auc_ovr, aps


def test_complete(loader, model, loss_fn, logger, phase, save_path):
    model.eval()
    running_corrects = 0
    running_loss = 0
    y_true = []
    y_pred = []
    y_probs = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            logits, proto_features, pooled = model(images.cuda(), inference=True)
            probs, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            p = F.softmax(logits, dim=1).data
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(p.cpu().numpy())

    # save y_pred and y_true in a csv file
    df = pd.DataFrame({'y_probs': y_probs,
                       'y_pred': y_pred,
                       'y_true': y_true})
    df.to_csv(f'{save_path}/results.csv', index=False)

    accuracy = running_corrects / len(loader.dataset)
    balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info('[INFO] {} acc, bacc , and loss: {}, {}, {}'.format(phase, accuracy, balanced_acc, loss))

    # draw the confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    classes = loader.dataset.classes
    if 'healthy' in classes:
        # replace 'healthy with 'normal'
        classes[classes.index('healthy')] = 'normal'

    if 'namd' in classes and len(classes) == 7:
        classes = ['ga', 'normal', 'iamd', 'namd', 'dme', 'rvo', 'stargardt']
    # uppercases all the classes
    classes = [i.upper() for i in classes]

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Rotate the x-axis labels
    plt.xticks(rotation=45)  # You can adjust the angle as needed

    # Add axis labels and title
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('')

    # Show plot
    plt.tight_layout()  # Optional: Ensure everything fits without overlap

    plt.savefig(save_path + '/conf_mtrx_simple.png', dpi=500)

    # classification report
    classification_report = metrics.classification_report(y_true=y_true, y_pred=y_pred,
                                                          target_names=classes, output_dict=True)
    logger.info('[INFO] classification report \n')
    logger.info(classification_report)

    #   calculate roc
    if len(classes) >= 2:
        calculate_roc(y_probs, y_true, logger)
        # calculate sensitivity and specificity
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        logger.info(f'f1: {f1}')

    return accuracy, balanced_acc, loss
