import os
import shutil
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import functional as F

from src.config import (CHAOS_REGION_ALIAS_TO_FILE_MAP,
                        CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                        CHAOS_REGION_ALIAS_TO_REGION_MAP,
                        MODEL_OUTPUT_PATH,
                        device)
from src.data_generator import DataGenerator
from src.utils.optuna_utility.evaluation import MaskRCNNOutputEvaluator
from src.model_objects.abstract_maskrcnn import AbstractMaskRCNN
from src.utils.file_utils import (make_dir, clear_and_remake_directory,
                                  create_output_csv, append_input_to_file)


class MaskRCNN(AbstractMaskRCNN):
    """
    Implementation of MaskRCNN class for chaos region analysis.
    """
    def __init__(self):
        """
        Initializes the MaskRCNN class, model output path,
        creates a CSV file for metrics, and initializes the CSV file.
        """
        super().__init__()
        clear_and_remake_directory(MODEL_OUTPUT_PATH)
        self.csv_file_name = "metrics.csv"
        self.csv_path = f"{MODEL_OUTPUT_PATH}/{self.csv_file_name}"
        self.init_csv(self.csv_path)

    def init_csv(self, csv_path):
        """
        Initializes the CSV file for storing metrics.

        Args:
            csv_path (str): Path to the CSV file.
        """
        header = f"chaos_region,crop_size,crop_size,f1,precision,recall,best_threshold,min_iou\n"
        create_output_csv(csv_path, header)

    def init_pixel_csv(self, csv_path):
        """
        Initializes the CSV file for pixel-level metrics.

        Args:
            csv_path (str): Path to the CSV file.
        """
        header = f"chaos_region,crop_size,crop_size,f1,precision,recall,best_threshold\n"
        create_output_csv(csv_path, header)

    def calc_and_save_region_metrics(self, region_alias, crop_size, min_iou=0.3):
        """
        Calculate and save metrics for a specific region. Gets the best threshold score
        for entire region, then calculates metrics. Metrics include F1 score,
        precision, recall. Saves metrics in CSV file.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            min_iou (float, optional): Minimum IoU threshold. Defaults to 0.3.
        """
        # Initialize data generator and evaluator
        data_gen = DataGenerator()
        eval_obj = MaskRCNNOutputEvaluator()

        # Load and prepare images and labels in chaos region
        # min_sheet_area is the minimum area for a sheet to be considered valid.
        imgs, lbls = data_gen.get_sweep_crops_and_prune_partial_labels(region_alias, crop_size, min_sheet_area=50)

        # Get best threshold for region
        best_threshold = self.get_best_f1_chaos_region_threshold(imgs, lbls, min_iou=min_iou)

        # Calculate metrics
        f1_scores = []
        precision_scores = []
        recall_scores = []

        # Iterate over images and labels
        for img, lbl in zip(imgs, lbls):
            img_tensor = self.convert_img_to_torch_tensor(img)
            # Get thresholded prediction
            pred = self.calc_thresholded_preds(img_tensor, best_threshold)

            # Update metrics lists
            f1_scores.append(eval_obj.calc_min_iou_f1(lbl, pred, min_iou=min_iou))
            precision_scores.append(eval_obj.calc_min_iou_precision(lbl, pred, min_iou=min_iou))
            recall_scores.append(eval_obj.calc_min_iou_recall(lbl, pred, min_iou=min_iou))

        # Calculate average metrics
        avg_f1 = sum(f1_scores)/len(f1_scores)
        avg_prec = sum(precision_scores)/len(precision_scores)
        avg_rec = sum(recall_scores)/len(recall_scores)

        # Save metrics to CSV
        obs = f"{region_alias},{crop_size},{crop_size},{avg_f1:.4f},{avg_prec:.4f},{avg_rec:.4f},{best_threshold:.4f},{min_iou:.4f}"
        append_input_to_file(self.csv_path, obs)

    def get_best_f1_chaos_region_threshold(self, imgs, lbls, min_iou=0.3):
        """
        Calculates the best threshold for F1 score for a chaos region by
        iterating over a range of thresholds and calculating the F1 score
        for each. Returns the threshold that gives the best average F1 score.

        Args:
            imgs (list): List of images.
            lbls (list): List of labels.
            min_iou (float, optional): Minimum IoU threshold. Defaults to 0.3.

        Returns:
            best_threshold (float): Best F1 threshold.
        """
        # Initialize evaluator and threshold range
        eval_obj = MaskRCNNOutputEvaluator()
        threshes = np.linspace(0.5, 1.0, 50)
        best_threshold, best_avg_f1 = 0, 0

        # Get best threshold
        for thresh in threshes:
            f1_scores = [eval_obj.calc_min_iou_f1(lbl,
                                                  self.calc_thresholded_preds(self.convert_img_to_torch_tensor(img), thresh),
                                                  min_iou=min_iou
                                                  ) for img, lbl in zip(imgs, lbls)]

            # Calculate average F1 score
            avg_f1 = np.mean(f1_scores)

            # Update best threshold
            if avg_f1 > best_avg_f1:
                best_avg_f1, best_threshold = avg_f1, thresh

        return best_threshold

    def get_dataset_thresh_sweeps(self,
                                  dataset,
                                  thresh_count,
                                  min_threshold=0.5,
                                  max_threshold=1.0
                                  ):
        """
        Get thresholded prediction sweeps for a dataset.

        Args:
            dataset: The input dataset.
            thresh_count (int): Number of thresholds in the sweep.
            min_threshold (float): Minimum threshold value.
            max_threshold (float): Maximum threshold value.

        Returns:
            thresh_sweeps (list): List of thresholded sweeps.
        """
        thresh_sweeps = []

        # Iterate over dataset
        for img, label_dict in dataset:
            label = label_dict["masks"]
            label = label[0, :, :].cpu().numpy()
            thresh_sweep = self.get_thresholded_pred_sweep(img,
                                                           thresh_count=thresh_count,
                                                           min_threshold=min_threshold,
                                                           max_threshold=max_threshold
                                                           )
            # Add to list
            thresh_pair = (thresh_sweep, label)
            thresh_sweeps.append(thresh_pair)

        return thresh_sweeps

    def get_thresholded_pred_sweep(self,
                                   img_tensor,
                                   thresh_count,
                                   min_threshold=0.5,
                                   max_threshold=1.0
                                   ):
        """
        Calculates model predictions for a range of thresholds
        for an image.

        Args:
            img_tensor: Input image tensor.
            thresh_count (int): Number of thresholds in the sweep.
            min_threshold (float): Minimum threshold value.
            max_threshold (float): Maximum threshold value.

        Returns:
            preds (list): List of thresholded predictions.
        """
        # Define thresholds and get predictions
        threshes = np.linspace(min_threshold, max_threshold, num=thresh_count)
        preds = [self.calc_thresholded_preds(img_tensor, thresh) for thresh in threshes]

        return preds

    def calc_thresholded_preds(self, img_tensor, threshold):
        """
        Calculates model predictions for an image for a single given
        threshold. Applies the threshold to the predicted probabilities
        to get the predicted labels.

        Args:
            img_tensor: Input image tensor.
            threshold (float): Threshold value.

        Returns:
            pred (numpy.ndarray): Thresholded prediction.
        """
        # Get predicted probabilities
        pred_probs = self.get_mask_probs(img_tensor)
        obj_count = pred_probs.shape[0]
        pred = np.zeros((pred_probs.shape[1], pred_probs.shape[2]))

        # Apply threshold to probabilities
        for ind in range(obj_count):
            obj_id = ind + 1
            pixel_prob = pred_probs[ind, :, :]
            pred_instance = np.where(pixel_prob >= threshold, obj_id, 0)
            pred = np.where(pred_instance > 0, pred_instance, pred)

        return pred

    def create_heatmap_overlay(self, orig_img, heatmap, alpha=0.5):
        """
        Create a heatmap overlay on the original image. Applies a red
        overlay to the areas of the image where the model predicts the
        presence of an object.

        Args:
            orig_img: Original image.
            heatmap: Heatmap to overlay.
            alpha (float): Alpha value for blending.

        Returns:
            overlay_img (numpy.ndarray): Overlay image.
        """
        heatmap_overlay = orig_img.copy()
        heatmap_overlay[:, :, 2] = heatmap
        heatmap_overlay[:, :, 0] = 0
        heatmap_overlay[:, :, 1] = 0
        # Blend images
        overlay_img = cv2.addWeighted(orig_img, 1 - alpha, heatmap_overlay, alpha, 0)
        brightness_factor = 1.5
        overlay_img = overlay_img * brightness_factor
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)

        return overlay_img

    def prepare_chaos_region_data(self, region_alias):
        """
        Prepare data for processing a chaos region.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            stride (int): Stride for scanning.

        Returns:
            tuple: Tuple containing img, lbl, img_sweep and region_crop.
        """
        # Extract relevant data for chaos region preparation
        img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias])
        lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])[:, :, 0]
        img_sweep = img.copy()
        img_sweep[:, :, 0] = np.where(lbl > 0, 255, img_sweep[:, :, 0]).astype(np.uint8)

        # Get Region Bounding Box
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])
        region_activation = np.where(region_lbl)
        region_expand = 64

        # Get crop iterations
        xmin = np.min(region_activation[1]) - region_expand
        xmax = np.max(region_activation[1]) + region_expand
        ymin = np.min(region_activation[0]) - region_expand
        ymax = np.max(region_activation[0]) + region_expand
        region_crop = [xmin, xmax, ymin, ymax]

        return img, lbl, img_sweep, region_crop

    def calc_logit_distribution_sweep(self,
                                      y_start,
                                      img,
                                      crop_size,
                                      stride,
                                      region_crop,
                                      x_iters,
                                      y_iters
                                      ):
        """
        Calculate logit distribution sweep for a region.

        Args:
            y_start (int): Starting y-coordinate for scanning.
            img: Input image.
            crop_size (int): Uniform height/width of each crop.
            stride (int): Stride for scanning.
            region_crop: Crop coordinates for the region.
            x_iters (int): Number of iterations in x-direction.
            y_iters (int): Number of iterations in y-direction.

        Returns:
            tuple: Tuple containing img_logit_dist (logit distribution) and
            img_count_dist (count distribution).
        """
        # Initialize prediction matrix
        img_logit_dist = np.zeros_like(img[:, :, 0]).astype(float)
        img_count_dist = np.zeros_like(img[:, :, 0]).astype(float)

        # Calculate logit distribution sweep
        for _ in range(y_iters):
            y_start += stride
            y_end = y_start + crop_size
            x_start = region_crop[0] - stride

            for _ in range(x_iters):
                x_start += stride
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                instance = np.zeros_like(img_crop)
                instance += 1
                img_crop = cv2.resize(img_crop, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
                img_tensor = self.convert_img_to_torch_tensor(img_crop)

                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                logit_dist = cv2.resize(logit_dist, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist
                img_count_dist[y_start:y_end, x_start:x_end] += instance

        # Normalize logit distribution
        img_count_dist = np.where(img_count_dist == 0, 1, img_count_dist)
        img_logit_dist = img_logit_dist / img_count_dist

        return img_logit_dist, img_count_dist

    def generate_rescaled_chaos_region_logit_scan(self,
                                                  region_alias,
                                                  crop_size,
                                                  file_name="logit_dist_scan"
                                                  ):
        """
        Generate a rescaled chaos region logit scan and save the result.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            file_name (str): Name of the output file.
        """
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop
        x_iters = int(((xmax - xmin) - crop_size) / stride) + 1
        y_iters = int(((ymax - ymin) - crop_size) / stride) + 1

        # Set minimum iterations
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride

        # Calculate logit distribution sweep
        img_logit_dist, img_count_dist = self.calculate_logit_distribution_sweep(y_start,
                                                                                 img,
                                                                                 crop_size,
                                                                                 stride,
                                                                                 region_crop,
                                                                                 x_iters,
                                                                                 y_iters
                                                                                 )

        # Normalize
        rescaled_logit_dist = self.calculate_and_normalize_logit_distribution(img_logit_dist, img_count_dist)

        # Create heatmap overlay and save
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)

    def get_rescaled_chaos_region_logit_scan(self,
                                             region_alias,
                                             crop_size,
                                             stride=64
                                             ):
        """
        Get a rescaled chaos region logit scan without saving the result.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            stride (int): Stride for scanning.

        Returns:
            rescaled_logit_dist (numpy.ndarray): Rescaled logit distribution.
        """
        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop
        x_iters = int(((xmax - xmin) - crop_size) / stride) + 1
        y_iters = int(((ymax - ymin) - crop_size) / stride) + 1

        # Set minimum iterations
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride

        # Calculate logit distribution sweep
        y_start = ymin - stride
        img_logit_dist, img_count_dist = self.calculate_logit_distribution_sweep(y_start,
                                                                                 img,
                                                                                 crop_size,
                                                                                 stride,
                                                                                 region_crop,
                                                                                 x_iters,
                                                                                 y_iters
                                                                                 )
        # Normalize
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

        return rescaled_logit_dist

    def generate_rescaled_chaos_region_prob_scan(self,
                                                    region_alias,
                                                    crop_size,
                                                    file_name="logit_dist_scan"
                                                    ):
        """
        Generate a rescaled chaos region probability scan and save the result.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            file_name (str): Name of the output file.
        """
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop
        x_iters = int(((xmax - xmin) - crop_size) / stride) + 1
        y_iters = int(((ymax - ymin) - crop_size) / stride) + 1

        # Set minimum iterations
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride

        # Calculate logit distribution sweep
        y_start = ymin - stride
        img_logit_dist, img_count_dist = self.calculate_logit_distribution_sweep(y_start,
                                                                                 img,
                                                                                 crop_size,
                                                                                 stride,
                                                                                 region_crop,
                                                                                 x_iters,
                                                                                 y_iters
                                                                                 )

        # Normalize
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create heatmap overlay and save
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)

    def generate_chaos_region_logit_scan(self,
                                         region_alias,
                                         crop_size,
                                         file_name="logit_dist_scan"
                                         ):
        """
        Generate a chaos region logit scan and save the result.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            file_name (str): Name of the output file.
        """
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"
        stride = 32

        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop
        x_iters = int(((xmax - xmin) - crop_size) / stride) + 1
        y_iters = int(((ymax - ymin) - crop_size) / stride) + 1

        # Set minimum iterations
        if y_iters < 1:
            y_iters = 1
        if x_iters < 1:
            x_iters = 1

        y_start = ymin - stride

        # Calculate logit distribution sweep
        y_start = ymin - stride
        img_logit_dist, img_count_dist = self.calculate_logit_distribution_sweep(y_start,
                                                                                 img,
                                                                                 crop_size,
                                                                                 stride,
                                                                                 region_crop,
                                                                                 x_iters,
                                                                                 y_iters
                                                                                 )

        # Normalize and threshold
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX)
        rescaled_logit_dist = np.where(rescaled_logit_dist > 40, 255, 0).astype(np.uint8)

        # Create heatmap overlay and save
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imwrite(file_path, heatmap)

    def generate_img_sweep(self,
                           img,
                           crop_size,
                           file_name="logit_dist_sweep"
                           ):
        """
        Generate a logit distribution sweep for an image and save the result.

        Args:
            img (numpy.ndarray): Input image.
            crop_size (int): Uniform height/width of each crop.
            file_name (str): Name of the output file.
        """
        file_path = f"{MODEL_OUTPUT_PATH}/{file_name}.png"

        # Initialize prediction matrix
        img_logit_dist = np.zeros_like(img[:, :, 0]).astype(float)
        img_sweep = img.copy()

        # Get image dimensions
        y_size = img.shape[0]
        x_size = img.shape[1]
        y_iters = int(y_size / crop_size)
        x_iters = int(x_size / crop_size)

        y_start = -crop_size

        # Calculate logit distribution sweep
        for _ in range(y_iters):
            y_start += crop_size
            y_end = y_start + crop_size
            x_start = -crop_size

            for _ in range(x_iters):
                x_start += crop_size
                x_end = x_start + crop_size

                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist

        # Normalize
        rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create heatmap overlay and save
        heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
        cv2.imwrite(file_path, heatmap)

    def generate_chaos_region_logit_sweep(self,
                                          region_alias,
                                          crop_size,
                                          file_name="logit_dist_sweep"
                                          ):
        """
        Generate a logit distribution sweep for a specific chaos region and save the result.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            file_name (str): Name of the output file.
        """
        file_path = f"{self.output_folder}/{file_name}.png"

        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop
        x_iters = int(((xmax - xmin) - crop_size)) + 1
        y_iters = int(((ymax - ymin) - crop_size)) + 1

        y_start = ymin - crop_size

        # Initialize prediction matrix
        img_logit_dist = np.zeros_like(img[:, :, 0]).astype(float)

        # Calculate logit distribution sweep
        for _ in range(y_iters):
            y_start += crop_size
            y_end = y_start + crop_size
            x_start = xmin - crop_size

            for _ in range(x_iters):
                x_start += crop_size
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                mask_logits = self.get_mask_logits(img_tensor)
                logit_dist = self.calc_logit_distribution_matrix(mask_logits)
                img_logit_dist[y_start:y_end, x_start:x_end] += logit_dist

            # Normalize
            rescaled_logit_dist = cv2.normalize(img_logit_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create heatmap overlay and save
            heatmap = self.create_heatmap_overlay(img_sweep, rescaled_logit_dist)
            cv2.rectangle(heatmap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imwrite(file_path, heatmap)

    def generate_chaos_region_thresh_sweep(self,
                                           region_alias,
                                           crop_size,
                                           thresh=0.52,
                                           file_name="logit_dist_sweep"
                                           ):
        """
        Generate a thresholded sweep for a specific chaos region.

        Args:
            region_alias (str): Alias of the chaos region.
            crop_size (int): Uniform height/width of each crop.
            thresh (float, optional): Threshold value. Defaults to 0.52.
            file_name (str, optional): Name of the output file. Defaults to "logit_dist_sweep".
        """
        file_path = f"{self.output_folder}/{file_name}.png"

        # Extract relevant data for chaos region preparation
        img, lbl, img_sweep, region_crop = self.prepare_chaos_region_data(region_alias)

        # Get Region Bounding Box
        xmin, xmax, ymin, ymax = region_crop

        y_start = ymin - crop_size

        # Initialize prediction matrix
        img_preds = np.zeros_like(img[:, :, 0]).astype(float)

        # Calculate thresholded sweep
        for _ in range(y_iters):
            y_start += crop_size
            y_end = y_start + crop_size
            x_start = xmin - crop_size

            for _ in range(x_iters):
                x_start += crop_size
                x_end = x_start + crop_size
                img_crop = img[y_start:y_end, x_start:x_end, 0].copy()
                img_tensor = self.convert_img_to_torch_tensor(img_crop)
                thresh_preds = self.calc_thresholded_preds(img_tensor, thresh)
                img_preds[y_start:y_end, x_start:x_end] += thresh_preds

        # Threshold predictions and save
        img_sweep[:, :, 2] = np.where(img_preds > 0, 255, img_sweep[:, :, 2]).astype(np.uint8)
        cv2.imwrite(file_path, img_sweep)

    def calc_logit_distribution_matrix(self, logits):
        """
        Calculates a distribution matrix for the logits
        (raw model outputs before applying the activation function).
        It sums up the logits for each pixel across all predicted
        instances.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            logit_dist (numpy.ndarray): Logit distribution matrix.
        """
        # Initialize prediction matrix
        logit_dist = np.zeros((logits.shape[2], logits.shape[3]))
        pred_instance_count = logits.shape[0]

        # Calculate logit distribution
        for pred_id in range(pred_instance_count):
            # Get prediction and convert to numpy array
            pred_np = logits[pred_id, 0, :, :].cpu().numpy()
            # Apply threshold
            pred_useful = np.where(pred_np > 0, 1, 0)
            logit_dist += pred_np

        return logit_dist

    def convert_img_to_torch_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            img_tensor (torch.Tensor): Converted image tensor.
        """
        img_tensor = F.to_tensor(img)
        return img_tensor

    def get_model_output(self, img_tensor):
        """
        Get the model output for a given image tensor.

        Args:
            img_tensor (torch.Tensor): Input image tensor.

        Returns:
            model_output (torch.Tensor): Model output.
        """
        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Set device and get model output
            model_output = self.model([img_tensor.to(device)])
        return model_output

    def get_mask_logits(self, img_tensor):
        """
        Gets the logits for the mask predictions from the model
        output for an image tensor.

        Args:
            img_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Mask logits.
        """
        model_output = self.get_model_output(img_tensor)
        # Get mask logits
        return model_output[0]['masks']

    def get_mask_probs(self, img_tensor):
        """
        Gets the probabilities for the mask predictions
        from the model output for an image tensor. It applies
        the sigmoid function to the logits to get the probabilities.

        Args:
            img_tensor (torch.Tensor): Input image tensor.

        Returns:
            mask_probs (numpy.ndarray): Mask probabilities.
        """
        model_output = self.get_model_output(img_tensor)
        # Get mask probabilities
        mask_probs = torch.sigmoid(model_output[0]['masks'][:, 0, :, :]).cpu().numpy()
        return mask_probs
