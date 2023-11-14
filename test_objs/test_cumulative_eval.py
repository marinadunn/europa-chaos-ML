from src.info.nasa_data import CHAOS_REGION_ALIAS_TO_FILE_MAP, CHAOS_REGION_ALIAS_TO_LABEL_MAP, CHAOS_REGION_ALIAS_TO_REGION_MAP
from src.model_objects.rcnn_model import MaskRCNN
from src.info.model_info import CUMULATIVE_TEST_MODEL_PATH
from src.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator
import cv2
import numpy as np
import matplotlib.pyplot as plt

class TestCumulativeEvaluator():
    # Testing object for the Evaluator class
    def __init__(self):
        self.test_region = "ii"
        self.region_lbl = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[self.test_region])
        region_activation = np.where(self.region_lbl)
        region_expand = 64
        self.x_min = np.min(region_activation[1]) - region_expand
        self.x_max = np.max(region_activation[1]) + region_expand
        self.y_min = np.min(region_activation[0]) - region_expand
        self.y_max = np.max(region_activation[0]) + region_expand
        x_min = np.min(region_activation[1]) - region_expand
        x_max = np.max(region_activation[1]) + region_expand
        y_min = np.min(region_activation[0]) - region_expand
        y_max = np.max(region_activation[0]) + region_expand
        self.base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[self.test_region])
        self.raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[self.test_region])
        self.base_crop = self.base_img[y_min:y_max, x_min:x_max, 0].copy()
        self.raw_label_crop = self.raw_label[y_min:y_max,x_min:x_max, :].copy()
        self.region_crop = self.region_lbl[y_min:y_max, x_min:x_max, 0].copy()
        self.lbl_crop = np.zeros_like(self.base_crop)
        self.base_thresh = 0.68 #.68
        self.min_iou = 0.3
        self.model_obj = MaskRCNN()
        self.model_obj.load_and_set_model(2, 2, model_path=CUMULATIVE_TEST_MODEL_PATH, type="v2")
        self.eval_obj = MaskRCNNCumulativeOutputEvaluator()
        self.process_label()
        self.img_tensor = self.model_obj.convert_img_to_torch_tensor(self.base_crop)
        self.vis_crop = self.base_img[y_min:y_max,x_min:x_max, :].copy()

    def process_label(self):
        self.lbl_crop = np.where(self.raw_label_crop[:,:,0] > 0, 1, 0)

    def run_tests(self):
        self.test_pixel_roc_curve()
        print("All Tests Passed")

    def test_pixel_roc_curve(self):
        pred_logit_dist = self.model_obj.get_rescaled_chaos_region_logit_scan(self.test_region, crop_size=384)
        pred_logit_dist = pred_logit_dist[self.y_min:self.y_max, self.x_min:self.x_max]
        pred_logit_dist = pred_logit_dist*self.region_crop
        auc, fprs, tprs = self.eval_obj.calc_pixel_auc(self.lbl_crop, pred_logit_dist, thresh_count=50, thresh_min=0, thresh_max=1)
        # print("AUC:", auc) auc def wrong
        # print("fprs", fprs)
        # print("tprs", tprs)
        plt.plot(fprs, tprs)
        plt.title("Pixel ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig("test_objs/basic_roc.png")
        plt.clf()
        pass

    def temp_save(self):
        pred_logit_dist = self.model_obj.get_rescaled_chaos_region_logit_scan(self.test_region, crop_size=384)
        pred_logit_dist = pred_logit_dist[self.y_min:self.y_max, self.x_min:self.x_max]
        pred_logit_dist = np.where(pred_logit_dist > 0.10, 255, 0)
        # pred_logit_dist = pred_logit_dist*self.region_crop
        self.vis_crop[:, :, 0] = np.where(self.lbl_crop > 0, 255, self.vis_crop[:, :, 0])
        self.vis_crop[:, :, 2] = np.where(pred_logit_dist > 0, 255, self.vis_crop[:, :, 2])
        cv2.imwrite("test_objs/test_case_cumulative_vis.png", self.vis_crop)