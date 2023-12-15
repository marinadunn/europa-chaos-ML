import cv2
import numpy as np
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from config import (TRANSFER_TRAINED_MODEL_FOLDER,
                    CV_OUTPUT_PATH,
                    CHAOS_REGION_ALIAS_TO_FILE_MAP,
                    CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                    CHAOS_REGION_ALIAS_TO_REGION_MAP
                    )
from model_objects.rcnn_model import MaskRCNN
from model_objects.trainable_maskrcnn import TrainableMaskRCNN
from utils.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator
from utils.file_utils import make_dir, clear_and_remake_directory, append_input_to_file, create_output_csv

# Create output directories and files
csv_path = f"{CV_OUTPUT_PATH}/pixel_metrics.csv"
header = f"chaos_region,crop_size,f1,precision,recall,best_threshold\n"
create_output_csv(csv_path, header)

# Constants
small_crop_size = 250
stride = 64
stride_eval = stride
min_thresh = 0.1
region_expand = 64
num_classes = 2
arch_id = 2  # Mask RCNN version 2
num_epochs = 15

def cross_val_pixel_csv():
    """
    Perform cross-validation and save pixel metrics to a CSV file.
    """
    regions = [ "A", "B", "C", "D",
               "E", "F", "G", "H", "I",
               "aa", "bb", "Co", "dd",
               "ee", "ff", "gg", "hh",
               "ii", "jj", "kk"
               ]

    start = 0
    end = len(regions)

    for i in range(start, end):
        test_region = regions[i]
        train_regions = [region for j, region in enumerate(regions) if j != i]
        train_obj = TrainableMaskRCNN(arch_id, num_classes)
        train_obj.train_model_given_regions(num_epochs,
                                            train_regions,
                                            [test_region],
                                            stride=stride,
                                            crop_size=small_crop_size
                                            )

        # Define model
        model = train_obj.get_model()
        model_obj = MaskRCNN()
        model_obj.set_model(model)
        # Evaluate model
        test_output(model_obj, test_region)

def crop_image(img, region_activation, region_expand):
    # Get crop bounds
    x_min = np.min(region_activation[1]) - region_expand
    x_max = np.max(region_activation[1]) + region_expand
    y_min = np.min(region_activation[0]) - region_expand
    y_max = np.max(region_activation[0]) + region_expand

    return img[y_min:y_max, x_min:x_max].copy()

def test_output(model_obj, region):
    """
    Evaluate model output on a specific region and save metrics and visualizations.
    """
    # Define evaluator
    eval_obj = MaskRCNNCumulativeOutputEvaluator()

    # Load images
    base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region])
    raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region])
    region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region])

    # Extract region of interest
    region_activation = np.where(region_lbl)
    region_expand = 64
    x_min = np.min(region_activation[1]) - region_expand
    x_max = np.max(region_activation[1]) + region_expand
    y_min = np.min(region_activation[0]) - region_expand
    y_max = np.max(region_activation[0]) + region_expand

    base_crop = base_img[y_min:y_max, x_min:x_max].copy()
    raw_label_crop = raw_label[y_min:y_max,x_min:x_max, :].copy()
    region_crop = region_lbl[y_min:y_max, x_min:x_max, 0].copy()

    # Preprocess labels
    lbl_crop = np.sum(raw_label_crop, axis=2)
    lbl_crop = np.where(lbl_crop > 0, 1, 0)
    region_crop = np.where(region_crop > 0, 1, 0)

    # Get model output
    pred_logit_dist = model_obj.get_rescaled_chaos_region_logit_scan(region,
                                                                     crop_size=small_crop_size,
                                                                     stride=stride_eval
                                                                     )
    pred_logit_dist = pred_logit_dist[y_min:y_max, x_min:x_max]

    # Remove out-of-region predictions
    pred_logit_dist = pred_logit_dist * region_crop

    # Get best threshold
    best_thresh = eval_obj.get_best_thresh_f1(lbl_crop,
                                              pred_logit_dist,
                                              thresh_count=100,
                                              thresh_min=min_thresh
                                              )

    # Calculate metrics
    tp, fp, tn, fn = eval_obj.calc_all_pixel_rates(lbl_crop,
                                                   pred_logit_dist,
                                                   threshold=best_thresh
                                                   )
    precision = eval_obj.calc_precision(tp, fp)
    recall = eval_obj.calc_recall(tp, fn)
    f1 = eval_obj.calc_f1_score(precision, recall)

    # Save metrics
    obs = f"{region},{small_crop_size},{f1:.4f},{precision:.4f},{recall:.4f},{best_thresh:.10f}"
    append_input_to_file(csv_path, obs)

    # Save visualizations
    thresholded_pred = np.where(pred_logit_dist > best_thresh, 255, 0)
    save_vis_performance(base_crop.copy(), thresholded_pred, lbl_crop, region)

    # Create ROC Curve
    roc_path = f"{CV_OUTPUT_PATH}/{region}_roc.png"
    auc, fprs, tprs = eval_obj.calc_pixel_auc(lbl_crop,
                                              pred_logit_dist,
                                              thresh_count=100,
                                              thresh_min=min_thresh
                                              )
    plt.plot(fprs, tprs)
    plt.title(f"Chaos {region} Pixel ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(roc_path)
    plt.clf()


def save_vis_performance(vis_crop, pred, true, region):
    """
    Save visualizations of model predictions and ground truth.
    """
    img_path = f"{CV_OUTPUT_PATH}/{region}_visualization.png"
    vis_crop[:, :, 0] = np.where(true > 0, 255, vis_crop[:, :, 0])
    vis_crop[:, :, 2] = np.where(pred > 0, 255, vis_crop[:, :, 2])
    cv2.imwrite(img_path, vis_crop)


if __name__ == "__main__":
    cross_val_pixel_csv()
