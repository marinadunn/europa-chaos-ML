import cv2
import numpy as np

from src.model_objects.rcnn_model import MaskRCNN
from src.config import (TRANSFER_TRAINED_MODEL_FOLDER,
                        CHAOS_REGION_ALIAS_TO_FILE_MAP,
                        CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                        CHAOS_REGION_ALIAS_TO_REGION_MAP,
                        MODEL_OUTPUT_PATH
                        )
from src.utils.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator
from src.utils.file_utils import create_output_csv, append_input_to_file

# Constants
arch_id = 2
region_expand = 64
thresh_count = 50
small_crop_size = 384

def pixel_csv():
    """
    Process images from specified regions, evaluate metrics, and save results to a CSV file.
    """
    # Load and set model
    model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/type_{arch_id}_transfer_trained_model.pth"
    model_obj = MaskRCNN()
    model_obj.load_and_set_model(arch_id, num_classes, type="v2", model_path=model_path)

    # Load evaluator
    eval_obj = MaskRCNNCumulativeOutputEvaluator()

    # Output CSV file setup
    csv_path = f"{MODEL_OUTPUT_PATH}/pixel_metrics.csv"
    header = f"chaos_region,crop_size,f1,precision,recall,best_threshold\n"
    create_output_csv(csv_path, header)

    # Test chaos regions
    test_regions = ["aa", "bb", "Co", "dd", "ee", "ff", "hh", "ii", "jj", "kk"]

    # Iterate through regions
    for region in test_regions:
        # Load images
        base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region])
        raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region])
        region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region])
        region_activation = np.where(region_lbl)

        # Get region extents
        x_min = np.min(region_activation[1]) - region_expand
        x_max = np.max(region_activation[1]) + region_expand
        y_min = np.min(region_activation[0]) - region_expand
        y_max = np.max(region_activation[0]) + region_expand

        # Extract region of interest
        base_crop = base_img[y_min:y_max, x_min:x_max, 0].copy()
        raw_label_crop = raw_label[y_min:y_max, x_min:x_max, :].copy()
        region_crop = region_lbl[y_min:y_max, x_min:x_max, 0].copy()

        # Preprocess labels
        lbl_crop = np.sum(raw_label_crop, axis=2)
        lbl_crop = np.where(lbl_crop > 0, 1, 0)
        region_crop = np.where(region_crop > 0, 1, 0)

        # Get model output
        pred_logit_dist = model_obj.get_rescaled_chaos_region_logit_scan(region, crop_size=small_crop_size)
        pred_logit_dist = pred_logit_dist[y_min:y_max, x_min:x_max]

        # Remove out-of-region predictions
        pred_logit_dist = pred_logit_dist * region_crop

        # Get best threshold for F1
        best_thresh = eval_obj.get_best_thresh_f1(lbl_crop, pred_logit_dist, thresh_count=thresh_count)

        # Calculate metrics for best threshold
        TP, FP, TN, FN = eval_obj.calc_all_pixel_rates(lbl_crop, pred_logit_dist, threshold=best_thresh)
        precision = eval_obj.calc_precision(TP, FP)
        recall = eval_obj.calc_recall(TP, FN)
        F1 = eval_obj.calc_f1_score(precision, recall)

        # Save metrics
        obs = f"{region}, {small_crop_size}, {F1:.4f}, {precision:.4f}, {recall:.4f}, {best_thresh:.4f}"
        append_input_to_file(csv_path, obs)


def utilize_trained_maskrcnn():
    """
    Utilize a trained MaskRCNN model to generate predictions for specific regions.
    """
    # Load model
    model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/type_{arch_id}_transfer_trained_model.pth"
    model_obj = MaskRCNN()
    model_obj.load_and_set_model(arch_id, num_classes, type="v2", model_path=model_path)

    # Define minimum iou score
    min_iou = 0.5

    # Get predictions for "Co" region
    model_obj.get_rescaled_chaos_region_logit_scan("Co", small_crop_size, file_name=f"Co_{small_crop_size}_logit_scan")

if __name__ == "__main__":
    pixel_csv()
