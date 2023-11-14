from src.model_objects.rcnn_model import MaskRCNN
from src.info.model_info import TRANSFER_TRAINED_MODEL_FOLDER
import cv2
from src.info.nasa_data import CHAOS_REGION_ALIAS_TO_FILE_MAP, CHAOS_REGION_ALIAS_TO_LABEL_MAP, CHAOS_REGION_ALIAS_TO_REGION_MAP
from src.info.file_structure import MODEL_OUTPUT_PATH
import numpy as np
from src.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator
import src.utility.file_text_processing as ftp


def pixel_csv():
    arch_id = 2
    model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/type_{arch_id}_transfer_trained_model.pth"
    model_obj = MaskRCNN()
    model_obj.load_and_set_model(arch_id, 2, type="v2", model_path=model_path)
    csv_path = f"{MODEL_OUTPUT_PATH}/pixel_metrics.csv"
    header = f"chaos_region,crop_size,f1,precision,recall,best_threshold\n"
    ftp.create_output_csv(csv_path, header)
    small_crop_size = 384
    eval_obj = MaskRCNNCumulativeOutputEvaluator()

    test_regions = ["aa", "bb", "Co", "dd", "ee", "ff", "hh", "ii", "jj", "kk"]
    for region in test_regions:
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
        base_crop = base_img[y_min:y_max, x_min:x_max, 0].copy()
        raw_label_crop = raw_label[y_min:y_max,x_min:x_max, :].copy()
        region_crop = region_lbl[y_min:y_max, x_min:x_max, 0].copy()
        # Preprocess
        lbl_crop = np.sum(raw_label_crop, axis=2)
        lbl_crop = np.where(lbl_crop > 0, 1, 0)
        region_crop = np.where(region_crop > 0, 1, 0)
        # Get model output
        pred_logit_dist = model_obj.get_rescaled_chaos_region_logit_scan(region, crop_size=small_crop_size)
        pred_logit_dist = pred_logit_dist[y_min:y_max, x_min:x_max]
        # REmove out of region preds
        pred_logit_dist = pred_logit_dist*region_crop
        # Get best Thresh
        best_thresh = eval_obj.get_best_thresh_f1(lbl_crop, pred_logit_dist, thresh_count=50)
        # Produce metrics
        tp, fp, tn, fn = eval_obj.calc_all_pixel_rates(lbl_crop, pred_logit_dist, threshold=best_thresh)
        prec = eval_obj.calc_precision(tp, fp)
        rec = eval_obj.calc_recall(tp, fn)
        f1 = eval_obj.calc_f1_score(prec, rec)
        # Save metrics
        obs = f"{region},{small_crop_size},{f1:.4f},{prec:.4f},{rec:.4f},{best_thresh:.4f}"
        ftp.append_input_to_file(csv_path, obs)


def utilize_trained_maskrcnn():
    arch_id = 2
    model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/type_{arch_id}_transfer_trained_model.pth"
    # region_hh_ii_file_path = 'data/c17ESNERTRM01_GalileoSSI_E_reproj.png'
    # region_Co_file_path = 'data/E6ESDRKLIN01_GalileoSSI_E_reproj.png'
    region_hh_ii_file_path = CHAOS_REGION_ALIAS_TO_FILE_MAP["hh"]
    region_Co_file_path = CHAOS_REGION_ALIAS_TO_FILE_MAP["Co"]
    region_hh_ii = cv2.imread(region_hh_ii_file_path)
    region_Co = cv2.imread(region_Co_file_path)
    model_obj = MaskRCNN()
    model_obj.load_and_set_model(arch_id, 2, type="v2", model_path=model_path)
    small_crop_size = 384
    large_crop_size = 300
    min_iou = 0.5


    model_obj.get_rescaled_chaos_region_logit_scan("Co", small_crop_size, file_name=f"Co_{small_crop_size}_logit_scan")


if __name__ == "__main__":
    pixel_csv()