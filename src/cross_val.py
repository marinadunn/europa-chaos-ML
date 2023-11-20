from config import (make_dir, clear_and_remake_directory,
                    TRANSFER_TRAINED_MODEL_FOLDER,
                    CV_OUTPUT_PATH,
                    CHAOS_REGION_ALIAS_TO_FILE_MAP,
                    CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                    CHAOS_REGION_ALIAS_TO_REGION_MAP
                    )
from model_objects.rcnn_model import MaskRCNN
from model_objects.trainable_maskrcnn import TrainableMaskRCNN
from utils.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator
import utils.file_utils as file_utils

import cv2
import numpy as np
import matplotlib.pyplot as plt


clear_and_remake_directory(CV_OUTPUT_PATH)
csv_path = f"{CV_OUTPUT_PATH}/pixel_metrics.csv"
header = f"chaos_region,crop_size,f1,precision,recall,best_threshold\n"
file_utils.create_output_csv(csv_path, header)
small_crop_size = 175
stride = 64
stride_eval = stride
mini_thresh=-2

def cross_val_pixel_csv():
    regions = ["Co", "A", "ii", "B", "C", "D", "ee", "F", "G", "hh", "I", "jj", "kk"]
    num_epochs = 13

    start = 0
    end = len(regions)

    for i in range(start,end):
        test_region = regions[i]
        train_regions = regions.copy()
        train_regions.pop(i)
        arch_id = 2
        train_obj = TrainableMaskRCNN(arch_id, 2)
        train_obj.train_model_given_regions(num_epochs, train_regions, [test_region], stride=stride, crop_size=small_crop_size)
        # train_obj.save_model()
        # del train_obj
        # model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/type_{arch_id}_transfer_trained_model.pth"
        model = train_obj.get_model()
        model_obj = MaskRCNN()
        model_obj.set_model(model)
        # model_obj.load_and_set_model(arch_id, 2, type="v2", model_path=model_path)
        test_output(model_obj, test_region)

def test_output(model_obj, region):
    eval_obj = MaskRCNNCumulativeOutputEvaluator()
    # Load
    base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[region])
    raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region])
    region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region])
    region_activation = np.where(region_lbl)
    region_expand = 64
    x_min = np.min(region_activation[1]) - region_expand
    x_max = np.max(region_activation[1]) + region_expand
    y_min = np.min(region_activation[0]) - region_expand
    y_max = np.max(region_activation[0]) + region_expand
    # Focus on Region
    base_crop = base_img[y_min:y_max, x_min:x_max].copy()
    raw_label_crop = raw_label[y_min:y_max,x_min:x_max, :].copy()
    region_crop = region_lbl[y_min:y_max, x_min:x_max, 0].copy()
    # Preprocess
    lbl_crop = np.sum(raw_label_crop, axis=2)
    lbl_crop = np.where(lbl_crop > 0, 1, 0)
    region_crop = np.where(region_crop > 0, 1, 0)
    # Get model output
    pred_logit_dist = model_obj.get_rescaled_chaos_region_logit_scan(region, crop_size=small_crop_size, stride=stride_eval)
    pred_logit_dist = pred_logit_dist[y_min:y_max, x_min:x_max]
    # REmove out of region preds
    pred_logit_dist = pred_logit_dist*region_crop
    # Get best Thresh
    best_thresh = eval_obj.get_best_thresh_f1(lbl_crop, pred_logit_dist, thresh_count=100, thresh_min=mini_thresh)
    # best_thresh = 0.15
    # best_thresh = eval_obj.get_best_thresh_precision(lbl_crop, pred_logit_dist, thresh_count=100, thresh_min=mini_thresh)
    # Produce metrics
    tp, fp, tn, fn = eval_obj.calc_all_pixel_rates(lbl_crop, pred_logit_dist, threshold=best_thresh)
    prec = eval_obj.calc_precision(tp, fp)
    rec = eval_obj.calc_recall(tp, fn)
    f1 = eval_obj.calc_f1_score(prec, rec)
    # Save metrics
    obs = f"{region},{small_crop_size},{f1:.4f},{prec:.4f},{rec:.4f},{best_thresh:.10f}"
    file_utils.append_input_to_file(csv_path, obs)
    # Save Vis
    thresholded_pred = np.where(pred_logit_dist > best_thresh, 255, 0)
    save_vis_performance(base_crop.copy(), thresholded_pred, lbl_crop, region)
    # Roc Curve
    roc_path = f"{CV_OUTPUT_PATH}/{region}_roc.png"
    auc, fprs, tprs = eval_obj.calc_pixel_auc(lbl_crop, pred_logit_dist, thresh_count=100, thresh_min=mini_thresh)
    plt.plot(fprs, tprs)
    plt.title(f"Chaos {region} Pixel ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(roc_path)
    plt.clf()


def save_vis_performance(vis_crop, pred, true, region):
    img_path = f"{CV_OUTPUT_PATH}/{region}_visualization.png"
    vis_crop[:, :, 0] = np.where(true > 0, 255, vis_crop[:, :, 0])
    vis_crop[:, :, 2] = np.where(pred > 0, 255, vis_crop[:, :, 2])
    cv2.imwrite(img_path, vis_crop)

if __name__ == "__main__":
    cross_val_pixel_csv()
