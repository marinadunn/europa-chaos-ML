from src.config import (CHAOS_REGION_ALIAS_TO_FILE_MAP,
                        CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                        TRANSFER_TRAINED_MODEL_PATH
                        )
from src.model_objects.rcnn_model import MaskRCNN
from src.utils.optuna_utility.evaluation import MaskRCNNOutputEvaluator

import cv2
import numpy as np
import matplotlib.pyplot as plt

# In test Image 8, true labels present
# Crop of interest sub_region = full_region[5202:5402, 1500:1700, 0].copy()

class TestEvaluator():
    """
    Testing object for the Evaluator class.
    """
    def __init__(self):
        """
        Initializes the TestEvaluator object.
        """
        y_min = 5195
        y_max = y_min + 200
        x_min = 1500
        x_max = x_min + 200

        # Load base image and raw label
        self.base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP["hh"])
        self.raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP["hh"])

        # Extract relevant crops
        self.base_crop = self.base_img[y_min:y_max, x_min:x_max, 0].copy()
        self.raw_label_crop = self.raw_label[y_min:y_max,x_min:x_max, :].copy()
        self.lbl_crop = np.zeros_like(self.base_crop)

        # Set base IoU threshold and minimum IoU threshold
        self.base_thresh = 0.68
        self.min_iou = 0.3

        # Load model
        self.model_obj = MaskRCNN()
        self.model_obj.load_and_set_model(1, 2, TRANSFER_TRAINED_MODEL_PATH)

        # Initialize evaluation object
        self.eval_obj = MaskRCNNOutputEvaluator()

        # Process label to convert RGB masks to integer labels
        self.process_label()

        # Convert base crop to torch tensor
        self.img_tensor = self.model_obj.convert_img_to_torch_tensor(self.base_crop)

        # Visualize crop
        self.vis_crop = self.base_img[y_min:y_max,x_min:x_max, :].copy()

    def process_label(self):
        """
        Process raw label to convert RGB masks to integer labels.
        """
        rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
        struc_data = self.raw_label_crop.view(rgb_type)
        unique_colors = np.unique(struc_data)
        mask_id = 1

        for void_color in unique_colors:
            mask_color = void_color.tolist()
            if mask_color == (0, 0, 0):
                continue
            single_mask = np.all(self.raw_label_crop == mask_color, axis=2)
            self.lbl_crop = np.where(single_mask > 0, mask_id, self.lbl_crop)
            mask_id += 1

    def run_tests(self):
        """
        Run a series of tests and assert the expected results.
        """
        self.test_basic_precision()
        self.test_basic_recall()
        self.test_min_iou_precision()
        self.test_min_iou_recall()
        self.test_min_iou_f1()
        print("All Tests Passed")

    def test_min_iou_precision(self):
        """
        Test minimum IoU precision metric.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        precision = self.eval_obj.calc_min_iou_precision(self.lbl_crop, thresh_pred, min_iou=self.min_iou)
        precision = round(precision, 3)
        assert precision == 0.333, "Min IoU Precision Metric Wrong"

    def test_min_iou_recall(self):
        """
        Test minimum IoU recall metric.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        recall = self.eval_obj.calc_min_iou_recall(self.lbl_crop, thresh_pred, min_iou=self.min_iou)
        recall = round(recall, 3)
        assert recall == 0.375, "Min IoU Recall Metric Wrong"

    def test_min_iou_f1(self):
        """
        Test minimum IoU F1 metric.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        F1 = self.eval_obj.calc_min_iou_f1(self.lbl_crop, thresh_pred, min_iou=self.min_iou)
        F1 = round(F1, 3)
        assert F1 == 0.353, "Min IoU F1 Metric Wrong"

    def test_min_iou_auc(self):
        """
        Test minimum IoU AUC metric.
        """
        thresh_sweep = self.model_obj.get_thresholded_pred_sweep(self.img_tensor, 50,
                                                                 min_threshold=0.5,
                                                                 max_threshold=1.0)
        auc, fpr, tpr = self.eval_obj.calc_min_iou_auc(self.lbl_crop,
                                                       thresh_sweep,
                                                       min_iou=self.min_iou)
        print("AUC:", auc)
        plt.plot(fpr, tpr)
        plt.savefig("min_iou_roc.png")
        plt.clf()

    def test_basic_precision(self):
        """
        Test basic precision metric.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        precision = self.eval_obj.calc_precision(self.lbl_crop, thresh_pred)
        precision = round(precision, 3)
        assert precision == 0.444, "Basic Precision Metric Wrong"

    def test_basic_recall(self):
        """
        Test basic recall metric.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        recall = self.eval_obj.calc_recall(self.lbl_crop, thresh_pred)
        recall = round(recall, 3)
        assert recall == 0.5, "Basic Recall Metric Wrong"

    def test_basic_auc(self):
        """
        Test basic AUC metric.
        """
        thresh_sweep = self.model_obj.get_thresholded_pred_sweep(self.img_tensor, 50,
                                                                 min_threshold=0.5,
                                                                 max_threshold=1.0)
        auc, precisions, recalls = self.eval_obj.calc_precision_recall_auc(self.lbl_crop, thresh_sweep)
        print("AUC:", auc)
        plt.plot(recalls, precisions)
        plt.savefig("basic_roc.png")
        plt.clf()

    def temp_save(self):
        """
        Save a visualization of the test case.
        """
        thresh_pred = self.model_obj.calc_thresholded_preds(self.img_tensor, self.base_thresh)
        self.vis_crop[:, :, 0] = np.where(self.lbl_crop > 0, 255, self.vis_crop[:, :, 0])
        self.vis_crop[:, :, 2] = np.where(thresh_pred > 0, 255, self.vis_crop[:, :, 2])
        cv2.imwrite("test_case_vis.png", self.vis_crop)
