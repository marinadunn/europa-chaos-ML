from src.config import (CHAOS_REGION_ALIAS_TO_FILE_MAP,
                        CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                        CHAOS_REGION_ALIAS_TO_REGION_MAP,
                        CUMULATIVE_TEST_MODEL_PATH
                        )
from src.model_objects.rcnn_model import MaskRCNN
from src.utils.optuna_utility.evaluation import MaskRCNNCumulativeOutputEvaluator

import cv2
import numpy as np
import matplotlib.pyplot as plt

class TestCumulativeEvaluator():
    """
    Testing object for the Evaluator class.
    """
    def __init__(self):
        self.test_region = "ii"
        self.setup_region_parameters()

        # Load base image and label
        self.base_img = cv2.imread(CHAOS_REGION_ALIAS_TO_FILE_MAP[self.test_region])
        self.raw_label = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[self.test_region])
        self.setup_crops()

        # Set up base and minimum IoU threshold
        self.base_thresh = 0.68
        self.min_iou = 0.3

        # Load model
        self.model_obj = MaskRCNN()
        self.model_obj.load_and_set_model(2, 2, model_path=CUMULATIVE_TEST_MODEL_PATH, type="v2")

        # Load evaluator
        self.eval_obj = MaskRCNNCumulativeOutputEvaluator()
        self.process_label()

        # Set up image tensor and visualization crop
        self.img_tensor = self.model_obj.convert_img_to_torch_tensor(self.base_crop)
        self.vis_crop = self.base_img[self.y_min:self.y_max, self.x_min:self.x_max, :].copy()

    def setup_region_parameters(self):
        """
        Set up region parameters.
        """
        # Get the region activation
        region_activation = np.where(cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[self.test_region]))
        region_expand = 64

        self.x_min = np.min(region_activation[1]) - region_expand
        self.x_max = np.max(region_activation[1]) + region_expand
        self.y_min = np.min(region_activation[0]) - region_expand
        self.y_max = np.max(region_activation[0]) + region_expand

    def setup_crops(self):
        """
        Set up image crops.
        """
        x_min = np.min(region_activation[1]) - region_expand
        x_max = np.max(region_activation[1]) + region_expand
        y_min = np.min(region_activation[0]) - region_expand
        y_max = np.max(region_activation[0]) + region_expand

        # Crop the base image and label
        self.base_crop = self.base_img[y_min:y_max, x_min:x_max, 0].copy()
        self.raw_label_crop = self.raw_label[y_min:y_max, x_min:x_max, :].copy()
        self.region_crop = np.where(self.raw_label_crop[:, :, 0] > 0, 1, 0)
        self.lbl_crop = np.zeros_like(self.base_crop)

    def process_label(self):
        """
        Process label.
        """
        self.lbl_crop = np.where(self.raw_label_crop[:, :, 0] > 0, 1, 0)

    def run_tests(self):
        """
        Run all tests.
        """
        self.test_pixel_roc_curve()
        print("All Tests Passed")

    def test_pixel_roc_curve(self):
        """
        Test pixel ROC curve.
        """
        # Get the predicted logit distribution
        pred_logit_dist = self.model_obj.get_rescaled_chaos_region_logit_scan(self.test_region, crop_size=384)
        pred_logit_dist = pred_logit_dist[self.y_min:self.y_max, self.x_min:self.x_max]
        pred_logit_dist = pred_logit_dist * self.region_crop

        # Calculate the pixel ROC curve
        auc, fpr, tpr = self.eval_obj.calc_pixel_auc(self.lbl_crop,
                                                     pred_logit_dist,
                                                     thresh_count=50,
                                                     thresh_min=0,
                                                     thresh_max=1
                                                     )

        # Plot the pixel ROC curve
        plt.plot(fpr, tpr)
        plt.title("Pixel ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig("basic_roc.png")
        plt.clf()

    def temp_save(self):
        """
        Temporarily save results.
        """
        # Get the predicted logit distribution
        pred_logit_dist = self.model_obj.get_rescaled_chaos_region_logit_scan(self.test_region, crop_size=384)
        pred_logit_dist = pred_logit_dist[self.y_min:self.y_max, self.x_min:self.x_max]
        pred_logit_dist = np.where(pred_logit_dist > 0.10, 255, 0)

        # Visualize the results and save
        self.vis_crop[:, :, 0] = np.where(self.lbl_crop > 0, 255, self.vis_crop[:, :, 0])
        self.vis_crop[:, :, 2] = np.where(pred_logit_dist > 0, 255, self.vis_crop[:, :, 2])
        cv2.imwrite("test_case_cumulative_vis.png", self.vis_crop)
