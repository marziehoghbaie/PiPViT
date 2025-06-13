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
from utils.model_utils import create_model

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
import cv2
from PIL import Image
import pandas as pd
def prep_mask(img, patches, prob):
    # apply softmax on each patch
    patches = [1 - patch for patch in patches]
    # this works the base for feature maps
    # Apply relu on the patches
    masks = [(mask.cpu().detach().numpy() - np.min(mask.cpu().detach().numpy())) / (
            np.max(mask.cpu().detach().numpy()) - np.min(mask.cpu().detach().numpy())) for mask in patches]
    hists = [np.histogram(mask, bins=7, range=(0, 1), density=True) for mask in masks]
    masks = [cv2.resize(mask, (img.shape[1], img.shape[0])) for mask in masks]
    img = np.float32(img) / 255
    heatmaps = [cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) for mask in masks]
    heatmaps = [np.float32(heatmap) / 255 for heatmap in heatmaps]
    cams = [heatmap + np.float32(img) for heatmap in heatmaps]
    cams = [(cam / np.max(cam)) for cam in cams]
    cams = [np.uint8(255 * cam) for cam in cams]
    return cams, hists


def main(config_path):
    print('Torch is available ', torch.cuda.is_available())

    (model_config, dataset_info, train_config, log_info, where, config,
     device, check_point_path, model_name, results_path, logger) = read_conf_file(config_path)
    dataset_info['shuffle'] = False
    train_config['batch_size'] = 1
    """ Create model """
    model = create_model(model_config, logger)
    torch.backends.cudnn.benchmark = False
    _ , _, _, transformation_test = transformers(model_config)
    # logger.info('[INFO] model: {}'.format(model))
    model = load_model(check_point_path, model, allow_size_mismatch=train_config['allow_size_mismatch'], device=device)
    logger.info(f'Model loaded from {check_point_path}')
    # set model on device
    model = model.to(device)
    model.eval()

    classes_dict = {'OCTDrusen': ['DRUSEN', 'NORMAL']}
    classes = classes_dict[dataset_info['dataset_name']]
    if dataset_info['dataset_name'] == 'OCTDrusen':
        sim_thresh = 1.
    else:
        sim_thresh = 0.3
    logger.info(f'Similarity Threshold : {sim_thresh}')

    # calculate sparsity
    subset = 'test'
    dataset_info['test_with_validation'] = False # test will be run on the test set
    logger.info(f'Loading test data from {dataset_info["data_path"]}/{subset}...')

    # create a list of all images
    images = []
    for cls in classes:
        imgs_list = [dataset_info['data_path']+f'/{subset}/{cls}/{img}' for img in os.listdir(dataset_info['data_path']+f'/{subset}/{cls}')]
        images.extend(imgs_list)

    logger.info(f'Number of images: {len(images)} from {subset} set...')
    model.eval()
    similarites_all = []

    for idx, img_path in enumerate(images):
        label = img_path.split(sep='/')[-2]

        img_ = Image.open(img_path).convert('RGB')
        img_ = transformation_test(img_)
        images = img_.unsqueeze(0)
        save_name = img_path.split(sep='/')[-1].split(sep='.')[0]
        out, blk_featrues, pooled = model(images.to(device), inference=True)
        # softmax over the output
        out = torch.nn.functional.softmax(out, dim=1)
        sorted_out, sorted_out_indices = torch.sort(out, descending=True)
        sorted_out_indices = sorted_out_indices.squeeze(0)
        pred_label = classes[sorted_out_indices[0]]
        logger.info('Sorted Prediction: {}{},pred_label  {} and label {}, name {}'.format(sorted_out_indices,sorted_out, pred_label, label, save_name))

        save_path = '/'.join([results_path, f'{label}_{pred_label}', save_name])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img)[:, :, ::-1]
        #
        blk_featrues = blk_featrues.squeeze(0) # [768, 12, 12]


        cams_heatmaps, hists = prep_mask(img, blk_featrues, None)
        imgs_heats = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in cams_heatmaps]
        pred_label = classes[sorted_out_indices[0]]

        os.makedirs(save_path, exist_ok=True)

        current_class = sorted_out_indices[0]
        # copy model head weights
        class_FC_weights = model._classification.weight[current_class].clone().detach().cpu().numpy()
        # save the weights
        # print non-zero weights in the class_FC_weights
        pooled_values = pooled.squeeze(0).clone().detach().cpu().numpy()
        similarities = class_FC_weights * pooled_values
        # convert similarities to a tensor
        similarities = torch.tensor(similarities)
        similarites_all.append(
            {'label': classes[current_class], 'img_name': save_name, 'similarities': similarities})

        # sort the similarities in descending order with their indexes
        sorted_sims, sorted_sims_indices = torch.sort(similarities, descending=True)
        for f_id, similarity in zip(sorted_sims_indices, sorted_sims):
            f_id = f_id.item()
            similarity = similarity.item()
            if similarity > sim_thresh:  # if the similarity is not zero
                name = f'heatmaps_blk_{pred_label}_{f_id}_{pooled_values[f_id]}_{similarity}_{class_FC_weights[f_id]}.png'
                fname = os.path.join(save_path, name)
                _heatmap = imgs_heats[f_id]
                # write the pooled values and the similarity values to the image
                cv2.putText(_heatmap, f'pooled: {pooled_values[f_id]:.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(_heatmap, f'similarity: {similarity:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.imwrite(fname, _heatmap)
        cv2.imwrite(os.path.join(save_path, 'org.png'), img)
    # save similarites_all as a dataframe
    similarites_all = pd.DataFrame(similarites_all)
    similarites_all.to_csv(f'{results_path}/similarites_all.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = args.config_path
    main(config_path)
