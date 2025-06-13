"""Dice score and average symmetric surface distance (ASSD). A higher Dice score and a lower ASSD value showcase better
segmentation performance."""



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


def prep_mask(img, patches):
    # apply softmax on each patch
    patches = [1 - patch for patch in patches]
    # this works the base for feature maps
    # Apply relu on the patches
    masks = [(mask.cpu().detach().numpy() - np.min(mask.cpu().detach().numpy())) / (
            np.max(mask.cpu().detach().numpy()) - np.min(mask.cpu().detach().numpy())) for mask in patches]
    # hists = [np.histogram(mask, bins=7, range=(0, 1), density=True) for mask in masks]
    masks = [cv2.resize(mask, (img.shape[1], img.shape[0])) for mask in masks]
    img = np.float32(img) / 255
    heatmaps_ = [cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) for mask in masks]
    heatmaps = [np.float32(heatmap) / 255 for heatmap in heatmaps_]
    cams = [heatmap + np.float32(img) for heatmap in heatmaps]
    cams = [(cam / np.max(cam)) for cam in cams]
    cams = [np.uint8(255 * cam) for cam in cams]
    return heatmaps_, cams

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    h_coor_min = h_idx * skip
    h_coor_max = min(img_size, h_idx * skip + patchsize)
    w_coor_min = w_idx * skip
    w_coor_max = min(img_size, w_idx * skip + patchsize)

    if h_idx == softmaxes_shape[1] - 1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# Function to convert a list of (x, y) coordinates
def convert_coordinates(coords, x_scale, y_scale):
    return [(x * x_scale, y * y_scale) for x, y in coords]


def perf(gt,pred,th):
    c = gt.sum()
    t = pred>=th
    cc = (gt*t).sum()
    area = t.sum()/np.prod(t.shape)
    return cc/c, area


def is_bbox_inside(bbox1, bbox2):
    """
    Check if bbox1 fits entirely within bbox2.

    bbox1 and bbox2 should be in the format [x_min, y_min, x_max, y_max].
    """
    return (bbox1[0] >= bbox2[0] and  # bbox1.x_min >= bbox2.x_min
            bbox1[1] >= bbox2[1] and  # bbox1.y_min >= bbox2.y_min
            bbox1[2] <= bbox2[2] and  # bbox1.x_max <= bbox2.x_max
            bbox1[3] <= bbox2[3])  # bbox1.y_max <= bbox2.y_max


def evaluation(model, image_paths, transformation_test, device,bbox_df, results_path, logger, scale, drusen_prototype_id):
    TPs = []  # correctly detected drusen
    FNs = []  # drusen that never get detected
    FPs = []  # flasely detected drusen, so the highlighted area does not have any drusen
    tp_color = (60,179,113) # green
    fp_color = (255,215,0) # Yellow
    fn_color = (255,99,71)  # Red
    color_detected_drusen = (153,50,204)  # violet
    # convert rgb to bgr
    tp_color = (tp_color[2], tp_color[1], tp_color[0])
    fp_color = (fp_color[2], fp_color[1], fp_color[0])
    fn_color = (fn_color[2], fn_color[1], fn_color[0])
    color_detected_drusen = (color_detected_drusen[2], color_detected_drusen[1], color_detected_drusen[0])
    for idx, img_path in enumerate(image_paths):
        TP = 0
        FN = 0
        FP = 0
        img_ = Image.open(img_path).convert('RGB')
        img_ = transformation_test(img_)
        images = img_.unsqueeze(0)
        save_name = img_path.split(sep='/')[-1].split(sep='.')[0]

        out, blk_featrues, pooled = model(images.to(device), inference=True)
        blk_featrues_ = blk_featrues.squeeze(0)  # [768, 12, 12]
        # this works the base for feature maps
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))
        img = np.array(img)[:, :, ::-1]
        heatmaps, heatmaps_with_image = prep_mask(img, blk_featrues_)


        predicted_map = heatmaps[drusen_prototype_id]
        gray = cv2.cvtColor(predicted_map, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        check_name = f'DRUSEN/{save_name}.jpeg'
        gt_bbox = bbox_df[bbox_df['image'] == check_name]

        heatmap_with_img = heatmaps_with_image[drusen_prototype_id]
        heatmap_with_img = cv2.cvtColor(heatmap_with_img, cv2.COLOR_BGR2RGB)

        for c_i, c in enumerate(cnts):
            fp_flag = True
            x, y, w, h = cv2.boundingRect(c)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Center the reduced box
            new_x = x + (w - new_w) // 2
            new_y = y + (h - new_h) // 2
            x, y, w, h = new_x, new_y, new_w, new_h
            detected = [x, y, x + w, y + h]

            for gt_drusen, gt in gt_bbox.iterrows():  # for each ground truth drusen we check to see if it falls into one the detected drusen areas
                gt_xmin, gt_ymin, gt_xmax, gt_ymax, gt_bbx_class = (gt['xmin'], gt['ymin'],
                                                                    gt['xmax'], gt['ymax'],
                                                                    gt['class'])

                cv2.rectangle(heatmap_with_img, (x, y), (x + w, y + h), color_detected_drusen, 2)
                if is_bbox_inside([gt_xmin, gt_ymin, gt_xmax, gt_ymax], detected):
                    fp_flag = False
                    TP += 1
                    cv2.rectangle(heatmap_with_img, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), tp_color, 2)

            if fp_flag:
                FP += 1
                cv2.rectangle(heatmap_with_img, (x, y), (x + w, y + h), fp_color, 2)

        for gt_drusen, gt in gt_bbox.iterrows():  # for each ground truth drusen we check to see if it falls into one the detected drusen areas
            gt_xmin, gt_ymin, gt_xmax, gt_ymax, gt_bbx_class = (gt['xmin'], gt['ymin'],
                                                                gt['xmax'], gt['ymax'],
                                                                gt['class'])
            fn_flag = True
            for c_i, c in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(c)
                new_w = int(w * scale)
                new_h = int(h * scale)
                # Center the reduced box
                new_x = x + (w - new_w) // 2
                new_y = y + (h - new_h) // 2
                x, y, w, h = new_x, new_y, new_w, new_h
                detected = [x, y, x + w, y + h]
                # check to see if the ground truth falls into the detected prototype area
                if is_bbox_inside([gt_xmin, gt_ymin, gt_xmax, gt_ymax], detected):
                    fn_flag = False
                    break
            if fn_flag:
                FN += 1
                cv2.rectangle(heatmap_with_img, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), fn_color, 2)

        # per Bscan
        FN /= len(gt_bbox)
        TP /= len(gt_bbox)
        FP /= len(gt_bbox)

        TPs.append(TP)
        FNs.append(FN)
        FPs.append(FP)
        # #write text on the image
        cv2.putText(heatmap_with_img, f'TP', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tp_color, 2)
        cv2.putText(heatmap_with_img, f'FP', (50, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fp_color, 2)
        cv2.putText(heatmap_with_img, f'FN', (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fn_color, 2)
        cv2.imwrite(f'{results_path}/{save_name}_detected.jpeg', heatmap_with_img)

    logger.info(f'Scale: {scale}, TPs: {np.mean(TPs)}, FPs: {np.mean(FPs)}, FNs: {np.mean(FNs)}')
    # calculate the precision, recall and F1 score
    precision = np.mean(TPs) / (np.mean(TPs) + np.mean(FPs))
    recall = np.mean(TPs) / (np.mean(TPs) + np.mean(FNs))
    F1 = 2 * (precision * recall) / (precision + recall)
    logger.info(f'Scale: {scale}, Precision: {precision}, Recall: {recall}, F1: {F1}')
    return precision, recall, F1, np.mean(TPs), np.mean(FPs), np.mean(FNs)



def main(config_path):
    """ Main function to visualize like pipnet"""
    print('Torch is available ', torch.cuda.is_available())

    (model_config, dataset_info, train_config, log_info, where, config,
     device, check_point_path, model_name, results_path, logger) = read_conf_file(config_path)
    dataset_info['shuffle'] = False
    train_config['batch_size'] = 1
    drusen_prototype_id = train_config['drusen_prototype_id']

    """ Create model """
    model = create_model(model_config, logger)
    torch.backends.cudnn.benchmark = False
    _ , _, _, transformation_test = transformers(model_config)
    logger.info('[INFO] model: {}'.format(model))
    model = load_model(check_point_path, model, allow_size_mismatch=False, device=device)
    logger.info(f'Model loaded from {check_point_path}')
    # set model on device
    model = model.to(device)
    model.eval()
    subset = 'Annotated'
    dataset_info['test_with_validation'] = False # test will be run on the test set
    annotation_path = dataset_info["annotation_path"]

    bbox_df = pd.read_csv(annotation_path)
    logger.info(f'Loading test data from {dataset_info["data_path"]}/{subset}...')
    # metrics for sparsity

    # create a list of all images
    image_paths = []
    for cls in ['DRUSEN']:
        imgs_list = [dataset_info['data_path']+f'/{subset}/{cls}/{img}' for img in os.listdir(dataset_info['data_path']+f'/{subset}/{cls}')]
        image_paths.extend(imgs_list)

    logger.info(f'Number of images: {len(image_paths)} from {subset} set...')
    model.eval()
    saved = dict()
    tensors_per_prototype = dict()

    for p in range(model._num_prototypes):
        saved[p]=0
        tensors_per_prototype[p]=[]

    logger.info(f'[INFO] {len(image_paths)} are processed ...')

    scales = np.arange(0.2, 10.1, 0.1)
    rows = []
    for scale in scales:
        results_path_scale = f'{results_path}/scale_{scale}'
        os.makedirs(results_path_scale, exist_ok=True)
        precision, recall, F1, meanTPs, meanFPs, meanFNs = evaluation(model, image_paths, transformation_test, device, bbox_df, results_path_scale, logger, scale, drusen_prototype_id)

        rows.append({'scale': scale, 'precision': precision, 'recall': recall,
                     'F1': F1, 'meanTPs': meanTPs, 'meanFPs': meanFPs, 'meanFNs': meanFNs})
    df = pd.DataFrame(rows)
    df.to_csv(f'{results_path}/evaluation_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = args.config_path
    main(config_path)