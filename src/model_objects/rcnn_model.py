
import os
import shutil 
from collections import Counter
from src.info.nasa_data import CHAOS_REGION_ALIAS_TO_FILE_MAP, CHAOS_REGION_ALIAS_TO_LABEL_MAP, CHAOS_REGION_ALIAS_TO_REGION_MAP
from src.info.file_structure import MODEL_OUTPUT_PATH
from src.data_generator import DataGenerator
from src.optuna_utility.evaluation import MaskRCNNOutputEvaluator
import numpy as np; #np.set_printoptions(threshold=np.inf,linewidth=np.inf)
from PIL import Image
import cv2

import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

import torchvision.transforms as T
from torchvision.transforms import functional as F
from src.model_objects.abstract_maskrcnn import AbstractMaskRCNN
from src.utility.file_system import make_dir, clear_and_remake_directory
import src.utility.file_text_processing as ftp
from src.info.nasa_data import CHAOS_REGION_ALIAS_TO_FILE_MAP, CHAOS_REGION_ALIAS_TO_REGION_MAP, CHAOS_REGION_ALIAS_TO_LABEL_MAP

class MaskRCNN(AbstractMaskRCNN):
    # Does anything that needs direct access to the model
    def __init__(self):
        super().__init__()
        # make_dir(MODEL_OUTPUT_PATH)
        clear_and_remake_directory(MODEL_OUTPUT_PATH)
        self.csv_file_name = "metrics.csv"
        self.csv_path = f"{MODEL_OUTPUT_PATH}/{self.csv_file_name}"
        self.init_csv(self.csv_path)

    def init_csv(self, csv_path):
        header = f"chaos_region,crop_height,crop_width,f1,precision,recall,best_threshold,min_iou\n"
        ftp.create_output_csv(csv_path, header)

    def init_pixel_csv(self, csv_path):
        header = f"chaos_region,crop_height,crop_width,f1,precision,recall,best_threshold\n"
        ftp.create_output_csv(csv_path, header)

    # Sweep through region only
    # Get best threshold for entire region
    # Get f1 score, precision and recall for entire region
    # Save in csv
    def calc_and_save_region_metrics(self, region_alias, crop_height, crop_width, min_iou=0.3):
        data_gen = DataGenerator()
        eval_obj = MaskRCNNOutputEvaluator()
        # Load and prepare imgs/labels in chaos region
        imgs, lbls = data_gen.get_sweep_crops_and_prune_partial_labels(
            region_alias,
            crop_height,
            crop_width,
            50)

        # Get best threshold for region
        best_threshold = self.get_best_f1_chaos_region_threshold(imgs, lbls, min_iou=min_iou)

        # Get metrics
        f1_scores = []
        precision_scores = []
        recall_scores = []
        for i in range(len(imgs)):
            img = imgs[i]
            lbl = lbls[i]
            img_tensor = self.convert_img_to_torch_tensor(img)
            pred = self.calc_thresholded_preds(img_tensor, best_threshold)
            f1_scores.append(eval_obj.calc_min_iou_f1(lbl, pred, min_iou=min_iou))
            precision_scores.append(eval_obj.calc_min_iou_precision(lbl, pred, min_iou=min_iou))
            recall_scores.append(eval_obj.calc_min_iou_recall(lbl, pred, min_iou=min_iou))
        avg_f1 = sum(f1_scores)/len(f1_scores)
        avg_prec = sum(precision_scores)/len(precision_scores)
        avg_rec = sum(recall_scores)/len(recall_scores)
        obs = f"{region_alias},{crop_height},{crop_width},{avg_f1:.4f},{avg_prec:.4f},{avg_rec:.4f},{best_threshold:.4f},{min_iou:.4f}"
        ftp.append_input_to_file(self.csv_path, obs)

    def get_best_f1_chaos_region_threshold(self, imgs, lbls, min_iou=0.3):
        eval_obj = MaskRCNNOutputEvaluator()
        threshes = np.linspace(0.5, 1.0, 50)
        best_threshold = 0
        best_avg_f1 = 0
        first = True
        for thresh in threshes:
            f1_scores = []
            for i in range(len(imgs)):
                img = imgs[i]
                lbl = lbls[i]
                img_tensor = self.convert_img_to_torch_tensor(img)
                pred = self.calc_thresholded_preds(img_tensor, thresh)
                f1_scores.append(eval_obj.calc_min_iou_f1(lbl, pred, min_iou=min_iou))
            avg_f1 = sum(f1_scores)/len(f1_scores)
            if first:
                first = False
                best_avg_f1 = avg_f1
                best_threshold = thresh
            else:
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_threshold = thresh
        return best_threshold

    def get_dataset_thresh_sweeps(self, dataset, thresh_count, min_threshold=0.5, max_threshold=1.0):
        thresh_sweeps = []

        for img, label_dict in dataset:
            label = label_dict["masks"]
            label = label[0,:,:].cpu().numpy()
            thresh_sweep = self.get_thresholded_pred_sweep(img, thresh_count=thresh_count, min_threshold=min_threshold, max_threshold=max_threshold)
            thresh_pair = (thresh_sweep, label)
            thresh_sweeps.append(thresh_pair)
        return thresh_sweeps

    def get_thresholded_pred_sweep(self, img_tensor, thresh_count, min_threshold=0.5, max_threshold=1.0):
        threshes = np.linspace(min_threshold, max_threshold, num=thresh_count)
        preds = []
        for thresh in threshes:
            pred = self.calc_thresholded_preds(img_tensor, thresh)
            preds.append(pred)
        return preds

    def calc_thresholded_preds(self, img_tensor, threshold):
        pred_probs = self.get_mask_probs(img_tensor)
        obj_count = pred_probs.shape[0]
        pred = np.zeros((pred_probs.shape[1], pred_probs.shape[2]))
        for ind in range(obj_count):
            obj_id = ind + 1
            pixel_prob = pred_probs[ind, :, :]
            pred_instance = np.where(pixel_prob >= threshold, obj_id, 0)
            pred = np.where(pred_instance > 0, pred_instance, pred) # Is this right?
        return pred

    def create_heatmap_overlay(self, orig_img, heatmap, alpha=0.5):
        heatmap_overlay = orig_img.copy()
        heatmap_overlay[:, :, 2] = heatmap
        heatmap_overlay[:, :, 0] = 0
        heatmap_overlay[:, :, 1] = 0
        overlay_img = cv2.addWeighted(orig_img, 1-alpha, heatmap_overlay, alpha, 0)
        brightness_factor = 1.5
        overlay_img = overlay_img*brightness_factor
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        return overlay_img

    def generate_rescaled_chaos_region_logit_scan(self, region_alias, crop_size, file_name="logit_dist_scan"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        region_expand = 64
        xmin = np.min(region_activation[1]) - region_expand
        xmax = np.max(region_activation[1]) + region_expand
        ymin = np.min(region_activation[0]) - region_expand
        ymax = np.max(region_activation[0]) + region_expand
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int(((xmax-xmin)-crop_size)/stride) + 1
        y_iters = int(((ymax-ymin)-crop_size)/stride) + 1
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later
        img_count_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += stride
            y_end = y_start + crop_size
            x_start = xmin - stride
            for col in range(x_iters):
                x_start += stride
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                instance = np.zeros_like(img_crop)
                instance += 1
                img_crop = cv2.resize(img_crop, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                # img_tensor = self.rescale_torch_tensor(img_tensor, self.train_size)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                logit_dist = cv2.resize(logit_dist, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist
                img_count_dist[y_start:y_end, x_start:x_end] += instance

        img_count_dist = np.where(img_count_dist == 0, 1, img_count_dist)
        img_logit_dist = img_logit_dist/img_count_dist
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        # rescaled_logit_dist = np.where(rescaled_logit_dist > 40, 255, 0)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)
        # file_path2 = f"{MODEL_OUTPUT_PATH}/{file_name}_test.png"
        # cv2.imwrite(file_path2, img_count_dist)

    def get_rescaled_chaos_region_logit_scan(self, region_alias, crop_size, stride=64):

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        region_expand = 64
        xmin = np.min(region_activation[1]) - region_expand
        xmax = np.max(region_activation[1]) + region_expand
        ymin = np.min(region_activation[0]) - region_expand
        ymax = np.max(region_activation[0]) + region_expand
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int(((xmax-xmin)-crop_size)/stride) + 1
        y_iters = int(((ymax-ymin)-crop_size)/stride) + 1
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later
        img_count_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += stride
            y_end = y_start + crop_size
            x_start = xmin - stride
            for col in range(x_iters):
                x_start += stride
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                instance = np.zeros_like(img_crop)
                instance += 1
                img_crop = cv2.resize(img_crop, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                # img_tensor = self.rescale_torch_tensor(img_tensor, self.train_size)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                logit_dist = cv2.resize(logit_dist, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4) # cv2.INTER_NEAREST
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist
                img_count_dist[y_start:y_end, x_start:x_end] += instance

        img_count_dist = np.where(img_count_dist == 0, 1, img_count_dist)
        img_logit_dist = img_logit_dist/img_count_dist
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return rescaled_logit_dist

    def generate_rescaled_chaos_region_prob_scan(self, region_alias, crop_size, file_name="logit_dist_scan"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        region_expand = 64
        xmin = np.min(region_activation[1]) - region_expand
        xmax = np.max(region_activation[1]) + region_expand
        ymin = np.min(region_activation[0]) - region_expand
        ymax = np.max(region_activation[0]) + region_expand
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int(((xmax-xmin)-crop_size)/stride) + 1
        y_iters = int(((ymax-ymin)-crop_size)/stride) + 1
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later
        img_count_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += stride
            y_end = y_start + crop_size
            x_start = xmin - stride
            for col in range(x_iters):
                x_start += stride
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                instance = np.zeros_like(img_crop)
                instance += 1
                img_crop = cv2.resize(img_crop, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                # img_tensor = self.rescale_torch_tensor(img_tensor, self.train_size)
                mask_logits = self.get_mask_probs(img_tensor)
                prob_dist = self.calc_prob_distribution_matrix(mask_logits)
                prob_dist = cv2.resize(prob_dist, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                img_logit_dist[y_start:y_end, x_start:x_end] += prob_dist
                img_count_dist[y_start:y_end, x_start:x_end] += instance

        img_count_dist = np.where(img_count_dist == 0, 1, img_count_dist)
        img_logit_dist = img_logit_dist/img_count_dist
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        # rescaled_logit_dist = np.where(rescaled_logit_dist > 40, 255, 0)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)
        # file_path2 = f"{MODEL_OUTPUT_PATH}/{file_name}_test.png"
        # cv2.imwrite(file_path2, img_count_dist)

    def generate_chaos_region_logit_scan(self, region_alias, crop_width,  crop_height, file_name="logit_dist_scan"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        region_expand = 64
        xmin = np.min(region_activation[1]) - region_expand
        xmax = np.max(region_activation[1]) + region_expand
        ymin = np.min(region_activation[0]) - region_expand
        ymax = np.max(region_activation[0]) + region_expand
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int(((xmax-xmin)-crop_width)/stride) + 1
        y_iters = int(((ymax-ymin)-crop_height)/stride) + 1
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later
        img_count_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += stride
            y_end = y_start + crop_height
            x_start = xmin - stride
            for col in range(x_iters):
                x_start += stride
                x_end = x_start + crop_width
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                instance = np.zeros_like(img_crop)
                instance += 1
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist
                img_count_dist[y_start:y_end, x_start:x_end] += instance

        img_count_dist = np.where(img_count_dist == 0, 1, img_count_dist)
        img_logit_dist = img_logit_dist/img_count_dist
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        rescaled_logit_dist = np.where(rescaled_logit_dist > 40, 255, 0)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)
        # file_path2 = f"{MODEL_OUTPUT_PATH}/{file_name}_test.png"
        # cv2.imwrite(file_path2, img_count_dist)

    def generate_img_sweep(self, img, crop_width,  crop_height, file_name="logit_dist_sweep"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later
        img_sweep = img.copy()

        y_size = img.shape[0]
        x_size = img.shape[1]
        y_iters = int(y_size/crop_height)
        x_iters = int(x_size/crop_width)

        y_start = -crop_height
        for row in range(y_iters):
            y_start += crop_height
            y_end = y_start + crop_height
            x_start = -crop_width
            for col in range(x_iters):
                x_start += crop_width
                x_end = x_start + crop_width
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist

        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.imwrite(file_path, heatmap)

    def generate_chaos_region_logit_sweep(self, region_alias, crop_width,  crop_height, file_name="logit_dist_sweep"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        xmin = np.min(region_activation[1])
        xmax = np.max(region_activation[1])
        ymin = np.min(region_activation[0])
        ymax = np.max(region_activation[0])
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int((xmax-xmin)/crop_width) + 1
        y_iters = int((ymax-ymin)/crop_height) + 1
        y_start = ymin - crop_height
        
        img_logit_dist = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += crop_height
            y_end = y_start + crop_height
            x_start = xmin - crop_width
            for col in range(x_iters):
                x_start += crop_width
                x_end = x_start + crop_width
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist

        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)

    def generate_chaos_region_thresh_sweep(self, region_alias, crop_width,  crop_height, thresh=0.52, file_name="logit_dist_sweep"):
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"

        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]) 
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:,:,0]
        img_sweep = img.copy()
        img_sweep[:,:,0] = np.where(lbl > 0, 255, img_sweep[:,:,0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        xmin = np.min(region_activation[1])
        xmax = np.max(region_activation[1])
        ymin = np.min(region_activation[0])
        ymax = np.max(region_activation[0])
        region_crop = [xmin, xmax, ymin, ymax]
        x_iters = int((xmax-xmin)/crop_width)
        y_iters = int((ymax-ymin)/crop_height)
        y_start = ymin - crop_height
        
        img_preds = np.zeros_like(img[:,:,0]).astype(float) #This is a silly way to do this change to np.zeros later

        for row in range(y_iters):
            y_start += crop_height
            y_end = y_start + crop_height
            x_start = xmin - crop_width
            for col in range(x_iters):
                x_start += crop_width
                x_end = x_start + crop_width
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                thresh_preds = self.calc_thresholded_preds(img_tensor, thresh)
                # thresh_preds = np.where(thresh_preds > 0 , 255, 0)
                img_preds[y_start:y_end, x_start:x_end] += thresh_preds

        img_sweep[:,:,2] = np.where(img_preds > 0, 255, img_sweep[:,:,2]).astype(np.uint8)
        cv2.imwrite(file_path, img_sweep)
    
    def generate_thresholded_dist_output_series(self):
        pass

    def generate_crop_threshold(self):
        pass

    def generate_dist_output(self, img, out_file_name):
        file_path = f"{self.output_folder}/{out_file_name}"
        output_img = img.copy()
        output_img = cv2.merge((output_img, output_img, output_img))
        mask_logits = self.get_mask_logits(img)
        logit_dist = self.calc_logit_distribution_matrix(mask_logits)

        rescaled_logit_dist = cv2.normalize(logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        rescaled_logit_dist = rescaled_logit_dist.astype(np.uint8)


        heatmap_overlay = output_img.copy()
        heatmap_overlay[:, :, 2] = rescaled_logit_dist
        heatmap_overlay[:, :, 0] = 0
        heatmap_overlay[:, :, 1] = 0
        alpha = 0.5
        output_img = cv2.addWeighted(output_img, 1-alpha, heatmap_overlay, alpha, 0)
        brightness_factor = 1.5
        output_img = output_img*brightness_factor
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    def calc_logit_distribution_matrix(self, logits):
        logit_dist = np.zeros((logits.shape[2], logits.shape[3]))
        pred_instance_count = logits.shape[0]
        for pred_id in range(pred_instance_count):
            pred_np = logits[pred_id, 0, :, :].cpu().numpy()
            pred_useful = np.where(pred_np > 0, 1, 0)
            logit_dist += pred_np
        return logit_dist

    def calc_prob_distribution_matrix(self, probs):
        prob_dist = np.zeros((probs.shape[1], probs.shape[2]))
        pred_instance_count = probs.shape[0]
        for pred_id in range(pred_instance_count):
            pred_np = probs[pred_id, :, :]
            prob_dist += pred_np
        return prob_dist

    def calc_cummulative_threshold_matrix(self, dataset, threshold=0.68):
        base_imgs = {}
        base_true_lbls = {}
        base_thresh_pred_lbls = {}
        dataset_type = dataset.dataset_type

        # Need to find all regions base img, label, and create a black pred canvas for them
        for dataset_sample in dataset:
            metadata = dataset_sample[2]
            region_alias = metadata[0]
            crop_metadata = metadata[1]
            if region_alias not in base_imgs.keys():
                base_img = np.array(Image.open(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:,:,0] 
                base_thresh_lbl = np.zeros_like(base_img).astype(float)
                base_img = cv2.merge((base_img, base_img, base_img))
                base_true_lbl = np.array(Image.open(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:,:,0] 

                base_imgs[region_alias] = base_img
                base_true_lbls[region_alias] = base_true_lbl.copy()
                base_thresh_pred_lbls[region_alias] = base_thresh_lbl

        for img_id, dataset_sample in enumerate(dataset):
            img = dataset_sample[0]
            metadata = dataset_sample[2]
            region_alias = metadata[0]
            crop_metadata = metadata[1]
            x_start = crop_metadata[0]
            x_end = crop_metadata[0] + crop_metadata[2]
            y_start = crop_metadata[1]
            y_end = crop_metadata[1] + crop_metadata[3]
            base_img = base_imgs[region_alias]
            base_thresh_pred_lbl = base_thresh_pred_lbls[region_alias]

            prediction = self.get_mask_probs(img)

            obj_count = prediction.shape[0]
            for obj_id in range(obj_count):
                pixel_prob = prediction[obj_id, :, :]
                pred_thresh_np = np.where(pixel_prob >= threshold, 1, 0)
                base_thresh_pred_lbl[y_start:y_end, x_start:x_end] += pred_thresh_np


        return base_true_lbls, base_thresh_pred_lbls

    def convert_img_to_torch_tensor(self, img):
        img_tensor = F.to_tensor(img)
        return img_tensor

    def rescale_torch_tensor(self, img_tensor, new_length):
        rescaled_tensor = F.resize(img_tensor, (new_length, new_length))
        return rescaled_tensor

    def get_model_output(self, img_tensor):
        # img_tensor = self.convert_img_to_torch_tensor(img)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model([img_tensor.to(self.device)])
        return model_output

    def get_mask_logits(self, img_tensor):
        model_output = self.get_model_output(img_tensor)
        return model_output[0]['masks']

    def get_mask_probs(self, img_tensor):
        model_output = self.get_model_output(img_tensor)
        mask_probs = torch.sigmoid(model_output[0]['masks'][:, 0, :, :]).cpu().numpy()
        return mask_probs