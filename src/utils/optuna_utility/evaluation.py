import numpy as np

class MaskRCNNCumulativeOutputEvaluator():
    def __init__(self):
        pass

    def calc_precision(self, tp, fp):
        """
        Calculate precision.

        Args:
            tp (int): True positives
            fp (int): False positives

        Returns:
            float: Precision value
        """
        return tp / max(tp + fp, 1)

    def calc_recall(self, tp, fn):
        """
        Calculate recall.

        Args:
            tp (int): True positives
            fn (int): False negatives

        Returns:
            float: Recall value
        """
        return tp / max(tp + fn, 1)

    def calc_f1_score(self, precision, recall):
        """
        Calculate F1 score.

        Args:
            precision (float): Precision value
            recall (float): Recall value

        Returns:
            float: F1 score
        """
        return (2 * precision * recall) / max(precision + recall, 1)

    def get_best_threshold(self, true, pred_logit_dist, metric_func,
                           thresh_count=20, thresh_min=0, thresh_max=1):
        """
        Find the best threshold based on a given metric.

        Args:
            true (numpy.ndarray): Ground truth
            pred_logit_dist (numpy.ndarray): Predicted logit distances
            metric_func (callable): Metric function to optimize
            thresh_count (int): Number of thresholds to consider
            thresh_min (float): Minimum threshold value
            thresh_max (float): Maximum threshold value

        Returns:
            float: Best threshold
        """
        threshes = np.logspace(thresh_min, thresh_max, num=thresh_count)

        # Calculate pixel rates for all thresholds in parallel using NumPy
        pixel_rates = np.array([
            self.calc_all_pixel_rates(true, pred_logit_dist, threshold=thresh)
            for thresh in threshes
        ])

        # Extract counts for true positives, false positives, true negatives, and false negatives
        tp, fp, tn, fn = pixel_rates.T

        # Calculate metric values for each threshold
        metric_values = metric_func(tp, fp, fn, tn)

        # Find the index of the maximum metric value
        best_index = np.argmax(metric_values)

        # Get the corresponding best threshold
        best_thresh = threshes[best_index]

        return best_thresh

    def get_best_thresh_f1(self, true, pred_logit_dist):
        """
        Find the best threshold based on F1 score.

        Args:
            true (numpy.ndarray): Ground truth
            pred_logit_dist (numpy.ndarray): Predicted logit distances
            kwargs: Additional parameters for get_best_threshold

        Returns:
            float: Best threshold
        """
        return self.get_best_threshold(true, pred_logit_dist, self.calc_f1_score, thresh_count, thresh_min, thresh_max)

    def get_best_thresh_precision(self, true, pred_logit_dist):
        """
        Find the best threshold based on precision.

        Args:
            true (numpy.ndarray): Ground truth
            pred_logit_dist (numpy.ndarray): Predicted logit distances
            kwargs: Additional parameters for get_best_threshold

        Returns:
            float: Best threshold
        """
        return self.get_best_threshold(true, pred_logit_dist, self.calc_precision, thresh_count, thresh_min, thresh_max)

    def calc_pixel_auc(self, true, pred_logit_dist, **kwargs):
        """
        Calculate area under the ROC curve for pixel-wise classification.

        Args:
            true (numpy.ndarray): Ground truth
            pred_logit_dist (numpy.ndarray): Predicted logit distances
            kwargs: Additional parameters for calc_auc

        Returns:
            tuple: AUC value, FPR, TPR
        """
        threshes = np.logspace(kwargs.get('thresh_min', 0),
                               kwargs.get('thresh_max', 1),
                               num=kwargs.get('thresh_count', 20)
                               )

        # Calculate pixel rates for each threshold in parallel
        pixel_rates = np.array([
            self.calc_all_pixel_rates(true, pred_logit_dist, threshold=thresh)
            for thresh in threshes
        ])

        # Extract counts for true positives, false positives, true negatives, and false negatives
        tp, fp, tn, fn = pixel_rates.T

         # Calculate false positive rate (FPR) and true positive rate (TPR)
        fpr = fp / np.maximum(fp + tn, 1)
        tpr = tp / np.maximum(tp + fn, 1)

        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)

        return auc, fpr, tpr

    def calc_all_pixel_rates(self, true, pred_logit_dist, threshold=0.5):
        """
        Calculate true positives, false positives, true negatives, and
        false negatives for pixel-wise classification.

        Args:
            true (numpy.ndarray): Ground truth
            pred_logit_dist (numpy.ndarray): Predicted logit distances
            threshold (float): Classification threshold

        Returns:
            tuple: True positives, false positives, true negatives, false negatives
        """
        # Binarize predictions based on the threshold
        pred_binary = (pred_logit_dist >= threshold).astype(int)

        # Create masks for true positive, false positive, true negative, and false negative
        true_positive_mask = (true == 1) & (pred_binary == 1)
        false_positive_mask = (true == 0) & (pred_binary == 1)
        true_negative_mask = (true == 0) & (pred_binary == 0)
        false_negative_mask = (true == 1) & (pred_binary == 0)

        # Calculate counts using masks
        tp = np.sum(true_positive_mask)
        fp = np.sum(false_positive_mask)
        tn = np.sum(true_negative_mask)
        fn = np.sum(false_negative_mask)

        return tp, fp, tn, fn

    def calc_auc(self, x_vals, y_vals):
        """
        Calculate area under the curve (AUC) using trapezoidal rule.

        Args:
            x_vals (list or numpy.ndarray): X-axis values
            y_vals (list or numpy.ndarray): Y-axis values

        Returns:
            float: AUC value
        """
        if len(x_vals) != len(y_vals):
            raise ValueError("Input arrays must have the same length.")

        # Sort x and y values based on x to ensure proper trapezoidal calculation
        sorted_indices = sorted(range(len(x_vals)), key=lambda k: x_vals[k])
        sorted_x = [x_vals[i] for i in sorted_indices]
        sorted_y = [y_vals[i] for i in sorted_indices]

        auc = 0

        # Iterate through the sorted values to calculate the trapezoidal areas
        for i in range(len(sorted_x) - 1):
            # Calculate the width and height of the trapezoid
            width = abs(sorted_x[i + 1] - sorted_x[i])
            average_height = (sorted_y[i] + sorted_y[i + 1]) / 2
            # Calculate the area of the trapezoid and add it to the total AUC
            sub_area = width * average_height
            auc += sub_area

        return auc


class MaskRCNNOutputEvaluator():
    """
    Provides metric evaluations for MaskRCNN output.
    """
    def __init__(self):
        pass

    def calc_min_iou_avg_f1(self, true, threshold_sweep, min_iou=0.5):
        """
        Calculate the average F1 score over a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds
            min_iou (float): Minimum IoU threshold

        Returns:
            avg_f1 (float): Average F1 score
            f1_scores (list): F1 scores for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        f1_scores = [self.calc_min_iou_f1(true, pred, min_iou=min_iou) for pred in threshold_sweep]
        # Average F1 score
        avg_f1 = sum(f1_scores) / len(f1_scores)
        return avg_f1, f1_scores

    def calc_min_iou_avg_precision(self, true, threshold_sweep, min_iou=0.5):
        """
        Calculate the average precision over a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds
            min_iou (float): Minimum IoU threshold

        Returns:
            avg_precision (float): Average precision
            precision_scores (list): Precision scores for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        precision_scores = [self.calc_min_iou_precision(true, pred, min_iou=min_iou) for pred in threshold_sweep]

        # Average precision
        avg_precision = sum(precision_scores) / len(precision_scores)

        return avg_precision, precision_scores

    def calc_min_iou_avg_recall(self, true, threshold_sweep, min_iou=0.5):
        """
        Calculate the average recall over a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds
            min_iou (float): Minimum IoU threshold

        Returns:
            avg_recall(float): Average recall
            recall_scores (list): Recall scores for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        recall_scores = [self.calc_min_iou_recall(true, pred, min_iou=min_iou) for pred in threshold_sweep]
        # Average recall
        avg_recall = sum(recall_scores) / len(recall_scores)
        return avg_recall, recall_scores

    def calc_min_iou_precision(self, true, pred, min_iou=0.5):
        """
        Calculate precision for a given minimum IoU threshold. If
        multiple test regions are provided, the average precision
        is returned.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output
            min_iou (float): Minimum IoU threshold

        Returns:
            float: Precision score
        """
        tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)

        # Avoid division by zero
        precision = tp / (tp + fp if (tp + fp) > 0 else 1)
        return precision

    def calc_min_iou_recall(self, true, pred, min_iou=0.5):
        """
        Calculate recall for a given minimum IoU threshold. If
        multiple test regions are provided, the average recall
        is returned.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output
            min_iou (float): Minimum IoU threshold

        Returns:
            float: Recall score
        """
        tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)
        recall = tp / max(1, (tp + fn))  # Use max to avoid division by zero
        return recall

    def calc_min_iou_f1(self, true, pred, min_iou=0.5):
        """
        Calculate F1 score for a given minimum IoU threshold. If
        multiple test regions are provided, the average F1 score
        is returned.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output
            min_iou (float): Minimum IoU threshold

        Returns:
            float: F1 score
        """
        # Calculate precision and recall
        precision = self.calc_min_iou_precision(true, pred, min_iou=min_iou)
        recall = self.calc_min_iou_recall(true, pred, min_iou=min_iou)

        # Calculate F1 score
        f1 = 0
        if (recall + precision) > 0:
            f1 = (2 * recall * precision) / (recall + precision)
        return f1

    def calc_min_iou_auc(self, true, threshold_sweep, min_iou=0.5):
        """
        Calculate the area under the curve (AUC) for a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds
            min_iou (float): Minimum IoU threshold

        Returns:
            auc (float): AUC value
            fprs (list): False positive rates for each threshold
            tprs (list): True positive rates for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        # Calculate FPRs and TPRs for each threshold
        fprs, tprs = zip(*[self.calc_all_rates(true, pred, min_iou=min_iou)[:2] for pred in threshold_sweep])
        # Calculate AUC
        auc = self.calc_auc(fprs, tprs)
        return auc, list(fprs), list(tprs)

    def calc_min_iou_precision_recall_auc(self, true, threshold_sweep, min_iou=0.5):
        """
        Calculate precision, recall, and AUC for a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds
            min_iou (float): Minimum IoU threshold

        Returns:
            auc (float): AUC value
            precisions (list): Precision values for each threshold
            recalls (list): Recall values for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        # Calculate precisions and recalls for each threshold
        precisions, recalls = zip(*[self.calc_all_rates(true, pred, min_iou=min_iou)[:2] for pred in threshold_sweep])
        # Calculate AUC
        auc = self.calc_auc(recalls, precisions)
        return auc, list(precisions), list(recalls)

    def calc_all_rates(self, true, pred, min_iou=0.5):
        """
        Calculate true positives, false positives, true negatives, and
        false negatives for a given minimum IoU threshold.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output
            min_iou (float): Minimum IoU threshold

        Returns:
            tuple: True positives, false positives, true negatives, false negatives
        """
        valid_preds = np.zeros_like(pred)

        tp = 0
        fp = self.get_class_count(pred)  # Assume all false positive for start
        tn = 1  # This is a placeholder, might not be sensical depending on the context
        fn = self.get_class_count(true)  # Assume all false negative for start

        # Iterate through each predicted instance
        for pred_id in np.unique(pred):
            if pred_id == 0:
                continue

            # Get the predicted instance
            pred_instance = np.where(pred == pred_id, 1, 0)
            pred_activated_true = true * pred_instance
            true_pred_overlap_count = len(np.unique(pred_activated_true))

            # Subtract 1 if there is a false positive
            if 0 in np.unique(pred_activated_true):
                true_pred_overlap_count -= 1

            # Only accept predictions that overlap one label
            if true_pred_overlap_count != 1:
                continue

            # Find the true label id that overlaps with the prediction
            true_id = np.unique(pred_activated_true)[1]
            true_instance = np.where(true == true_id, 1, 0)

            iou = self.calc_simple_iou_score(true_instance, pred_instance)

            # Update counts based on IoU threshold
            if iou > min_iou:
                tp += 1
                fp -= 1

        # Update counts based on IoU threshold
        fn -= tp
        return tp, fp, tn, fn

    def get_class_count(self, segms):
        """
        Gets the number of unique classes in a segmentation map.

        Args:
            segms (numpy.ndarray): Segmentation map

        Returns:
            int: Number of unique classes in the segmentation map
        """
        count = len(np.unique(segms))
        # Subtract 1 if there is a false positive
        if 0 in np.unique(segms):
            count -= 1
        return count

    def get_valid_preds(self, true, pred, min_iou=0.5):
        """
        Get valid predictions based on a minimum Intersection over Union (IoU) threshold.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output
            min_iou (float, optional): Minimum IoU threshold. Defaults to 0.5.

        Returns:
            numpy.ndarray: Valid predictions after applying the IoU threshold
        """
        valid_preds = np.zeros_like(pred)

        # Iterate through each predicted instance
        for pred_id in np.unique(pred):
            if pred_id == 0:
                continue

            # Get the predicted instance
            pred_instance = np.where(pred == pred_id, 1, 0)
            pred_activated_true = true * pred_instance
            true_pred_overlap_count = len(np.unique(pred_activated_true))

            # Subtract 1 if there is a false positive
            if 0 in np.unique(pred_activated_true):
                true_pred_overlap_count -= 1

            # Only accept predictions that overlap one label
            if true_pred_overlap_count != 1:
                continue

            # Get the true instance
            true_id = np.unique(pred_activated_true)[1]
            true_instance = np.where(true == true_id, 1, 0)

            # Calculate IoU
            iou = self.calc_simple_iou_score(true_instance, pred_instance)

            # Only accept predictions with IoU greater than the threshold
            if iou > min_iou:
                valid_preds += pred_instance

        return valid_preds

    def calc_f1_score(self, precision, recall):
        """
        Calculate the F1 score.

        Args:
            precision (float): Precision value
            recall (float): Recall value

        Returns:
            f1_score (float): F1 score
        """
        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score

    def calc_cumulative_threshold_iou(self, true_segms, pred_segms):
        """
        Calculate the cumulative threshold Intersection over Union (IoU) score.

        Args:
            true_segms (dict): Dictionary of ground truth segmentation masks
            pred_segms (dict): Dictionary of predicted segmentation masks

        Returns:
            float: Mean IoU score over all regions
        """
        iou_scores = [self.calc_simple_iou_score(np.where(true_segms[region] > 0, 1, 0),
                                                 np.where(pred_segms[region] > 0, 1, 0))
                      for region in true_segms.keys()]
        return np.mean(iou_scores)

    def calc_precision(self, true, pred):
        """
        Calculate the precision.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output

        Returns:
            precision (float): Precision value
        """
        true_activation = (true > 0).astype(int)
        true_activated_pred = pred * true_activation

        # Count the true positive instances
        label_count = np.count_nonzero(true_activated_pred)

        # If there is a false positive, subtract it
        if 0 in true_activated_pred:
            label_count -= 1

        # Count the total predicted instances
        unique_pred, counts_pred = np.unique(pred, return_counts=True)
        total_pred = np.sum(unique_pred[1:] * counts_pred[1:])  # exclude background class

        precision = -1 if total_pred == 0 else label_count / total_pred
        return precision

    def calc_recall(self, true, pred):
        """
        Calculate the recall.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output

        Returns:
            recall (float): Recall value
        """
        true_activation = (true > 0).astype(int)
        true_activated_pred = pred * true_activation

        label_count = np.count_nonzero(true_activated_pred)

        # Subtract 1 if there is a false negative
        label_count -= 1 if 0 in true_activated_pred else 0

        total_true = len(np.unique(true)) - (1 if 0 in np.unique(true) else 0)

        # Calculate recall; return 0 if there are no true instances
        recall = label_count / total_true if total_true != 0 else 0

        return recall

    def calc_tpr(self, true, pred):
        """
        Calculate the True Positive Rate (TPR).

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output

        Returns:
            tpr (float): True Positive Rate
        """
        tpr = self.calc_recall(true, pred)
        return tpr

    def calc_fpr(self, true, pred):
        """
        Calculate the False Positive Rate (FPR).

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output

        Returns:
            fpr (float): False Positive Rate
        """
        pass  # Implement FPR calculation if needed

    def calc_precision_recall_auc(self, true, threshold_sweep):
        """
        Calculate precision, recall, and AUC for a range of thresholds.

        Args:
            true (numpy.ndarray): Ground truth
            threshold_sweep (list): List of predicted outputs for different thresholds

        Returns:
            auc (float): AUC value
            precisions (list): Precision values for each threshold
            recalls (list): Recall values for each threshold
        """
        if len(threshold_sweep) < 2:
            raise ValueError("Not enough threshold output. Need 2 or more")

        precisions, recalls = zip(*[(self.calc_precision(true, pred), self.calc_recall(true, pred))
                                    for pred in threshold_sweep if self.calc_precision(true, pred) != -1])
        # Calculate AUC
        auc = self.calc_auc(recalls, precisions)
        return auc, list(precisions), list(recalls)

    def calc_auc(self, x_vals, y_vals):
        """
        Calculate the area under the curve (AUC) using the
        trapezoidal rule.

        Args:
            x_vals (list or numpy.ndarray): X-axis values
            y_vals (list or numpy.ndarray): Y-axis values

        Returns:
            auc (float): AUC value
        """
        x_vals, y_vals = np.array(x_vals), np.array(y_vals)
        # Sort x and y values by x
        sorted_indices = np.argsort(x_vals)
        x_vals, y_vals = x_vals[sorted_indices], y_vals[sorted_indices]
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(y_vals, x_vals)
        return auc

    def calc_simple_iou_score(self, true, pred):
        """
        Calculate the Intersection over Union (IoU) score.

        Args:
            true (numpy.ndarray): Ground truth
            pred (numpy.ndarray): Predicted output

        Returns:
            iou(float): IoU score
        """
        # Intersection is the overlapping area
        intersection = np.logical_and(true, pred)
        intersection_score = np.sum(intersection)

        # Union is the total area minus the intersection
        union = np.logical_or(true, pred)
        union_score = np.sum(union)

        # If the union is empty, the IoU is defined as 0
        if union_score == 0:
            iou = 0.0
        else:
            iou = intersection_score / union_score

        return iou
