import numpy as np
import cv2

class MaskRCNNCumulativeOutputEvaluator():
    def __init__(self):
        pass

    def calc_precision(self, tp, fp):
        if (tp+fp) == 0:
            return 0
        else:
            return tp/(tp+fp)

    def calc_recall(self, tp, fn):
        if (tp+fn) == 0:
            return 0
        else:
            return tp/(tp+fn)

    def calc_f1_score(self, precision, recall):
        if (recall + precision) == 0:
            return 0
        else:
            return (2*recall*precision)/(recall+precision)

    def get_best_thresh_f1(self, true, pred_logit_dist, thresh_count=20, thresh_min=-10, thresh_max=0):
        threshes = np.logspace(thresh_min, thresh_max, num=thresh_count)
        first = True
        best_f1 = 0
        best_thresh = 0
        for thresh in threshes:
            tp, fp, tn, fn = self.calc_all_pixel_rates(true, pred_logit_dist, threshold=thresh)
            prec = self.calc_precision(tp, fp)
            rec = self.calc_recall(tp, fn)
            f1 = self.calc_f1_score(prec, rec)
            if first:
                first = False
                best_f1 = f1
                best_thresh = thresh
            else:
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        return best_thresh

    def get_best_thresh_precision(self, true, pred_logit_dist, thresh_count=20, thresh_min=-10, thresh_max=0):
        threshes = np.logspace(thresh_min, thresh_max, num=thresh_count)
        first = True
        best_prec = 0
        best_thresh = 0
        for thresh in threshes:
            tp, fp, tn, fn = self.calc_all_pixel_rates(true, pred_logit_dist, threshold=thresh)
            prec = self.calc_precision(tp, fp)
            if first:
                first = False
                best_prec = prec
                best_thresh = thresh
            else:
                if prec > best_prec:
                    best_prec = prec
                    best_thresh = thresh
        return best_thresh

    def calc_pixel_auc(self, true, pred_logit_dist, thresh_count=20, thresh_min=-10, thresh_max=0):
        threshes = np.logspace(thresh_min, thresh_max, num=thresh_count)
        fprs = []
        tprs = []
        for thresh in threshes:
            tp, fp, tn, fn = self.calc_all_pixel_rates(true, pred_logit_dist, threshold=thresh)
            fpr = fp/(fp+tn)
            tpr = tp/(tp+fn)
            fprs.append(fpr)
            tprs.append(tpr)
        auc = self.calc_auc(fprs, tprs)
        return auc, fprs, tprs

    def calc_all_pixel_rates(self, true, pred_logit_dist, threshold=0.5):
        x_dim = true.shape[1]
        y_dim = true.shape[0]
        pred = np.where(pred_logit_dist >= threshold, 1, 0)
        true_null = np.where(true == 0, 1, 0)
        pred_null = np.where(pred_logit_dist < threshold, 1, 0)

        tp = np.sum(true*pred)
        tn = np.sum(true_null*pred_null)
        fn = np.sum(pred_null) - tn
        fp = np.sum(pred) - tp
        return tp, fp, tn, fn

    def calc_auc(self, x_vals, y_vals):
        # This is an imprecise method
        auc = 0
        for i in range(len(y_vals)-1):
            curr_y = y_vals[i]
            curr_x = x_vals[i]
            next_y = y_vals[i+1]
            next_x = x_vals[i+1]
            width = abs(curr_x - next_x)
            height = abs(next_y - curr_y)
            sub_area = width*height
            auc += sub_area
        return auc

class MaskRCNNOutputEvaluator():
    # Works with the MaskRCNN object to provide metric evaluations of its output
    def __init__(self):
        pass
    def calc_min_iou_avg_f1(self, true, threshold_sweep, min_iou=0.5):
        if len(threshold_sweep) < 2:
            print("Not enough treshold output. Need 2 or more") #Make exception later
        f1_scores = []
        for pred in threshold_sweep:
            f1_score = self.calc_min_iou_f1(true, pred, min_iou=min_iou)
            f1_scores.append(f1_score)
        avg_f1 = sum(f1_scores)/len(f1_scores)
        return avg_f1, f1_scores
    def calc_min_iou_avg_recall(self, true, threshold_sweep, min_iou=0.5):
        if len(threshold_sweep) < 2:
            print("Not enough treshold output. Need 2 or more") #Make exception later
        recall_scores = []
        for pred in threshold_sweep:
            recall_score = self.calc_min_iou_recall(true, pred, min_iou=min_iou)
            recall_scores.append(recall_score)
        avg_recall = sum(recall_scores)/len(recall_scores)
        return avg_recall, recall_scores
    # If multiple test regions does the avg of them
    def calc_min_iou_precision(self, true, pred, min_iou=0.5):
        tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)
        precision = 0
        if (tp+fp) > 0:
            precision = tp/(tp+fp)
        return precision
    def calc_min_iou_recall(self, true, pred, min_iou=0.5):
        tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)
        recall = 0
        if (tp+fn) > 0:
            recall = tp/(tp+fn)
        return recall
    def calc_min_iou_f1(self, true, pred, min_iou=0.5):
        precision = self.calc_min_iou_precision(true, pred, min_iou=min_iou)
        recall = self.calc_min_iou_recall(true, pred, min_iou=min_iou)
        f1 = 0
        if (recall+precision) > 0:
            f1 = (2*recall*precision)/(recall+precision)
        return f1
    def calc_min_iou_auc(self, true, threshold_sweep, min_iou=0.5):
        if len(threshold_sweep) < 2:
            print("Not enough treshold output. Need 2 or more") #Make exception later
        fprs = []
        tprs = []
        for pred in threshold_sweep:
            tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)
            fpr = fp/(fp+tn)
            tpr = tp/(tp+fn)
            fprs.append(fpr)
            tprs.append(tpr)
        auc = self.calc_auc(fprs, tprs)
        return auc, fprs, tprs
    def calc_min_iou_precision_recall_auc(self, true, threshold_sweep, min_iou=0.5):
        # NOT COMPLETE YET
        if len(threshold_sweep) < 2:
            print("Not enough treshold output. Need 2 or more") #Make exception later
        precisions = []
        recalls = []
        for pred in threshold_sweep:
            tp, fp, tn, fn = self.calc_all_rates(true, pred, min_iou=min_iou)
            precision = -1
            if (tp+fp) != 0:
                precision = tp/(tp+fp)
            if precision != -1:
                recall = tp/(tp+fn)
                precisions.append(precision)
                recalls.append(recall)
        auc = self.calc_auc(recalls, precisions)
        return auc, precisions, recalls
    def calc_all_rates(self, true, pred, min_iou=0.5):
        valid_preds = np.zeros_like(pred)
        tp = 0
        fp = self.get_class_count(pred) # Assume all false positive for start
        tn = 1 # Is this even sensical?
        fn = self.get_class_count(true) # Assume all false negative for start
        for pred_id in np.unique(pred):
            if pred_id == 0:
                continue
            pred_instance = np.where(pred == pred_id, 1, 0)
            pred_activated_true = true*pred_instance
            true_pred_overlap_count = len(np.unique(pred_activated_true))
            if 0 in np.unique(pred_activated_true):
                true_pred_overlap_count -= 1
            if true_pred_overlap_count != 1: # Only accept preds that overlap one label
                continue

            # We want the id that is nonzero, list can only have 1-2 ids within it
            true_id = -1
            for possible_id in np.unique(pred_activated_true):
                if possible_id == 0:
                    continue
                true_id = possible_id
                break
            true_instance = np.where(true == true_id, 1, 0)
            iou = self.calc_simple_iou_score(true_instance, pred_instance)
            if iou > min_iou:
                tp += 1
                fp -= 1

        fn -= tp
        return tp, fp, tn, fn

    def get_class_count(self, segs):
        count = len(np.unique(segs))
        if 0 in np.unique(segs):
            count -= 1
        return count

    def get_valid_preds(self, true, pred, min_iou=0.5):
        valid_preds = np.zeros_like(pred)
        for pred_id in np.unique(pred):
            if pred_id == 0:
                continue
            pred_instance = np.where(pred == pred_id, 1, 0)
            pred_activated_true = true*pred_instance
            true_pred_overlap_count = len(np.unique(pred_activated_true))
            if 0 in np.unique(pred_activated_true):
                true_pred_overlap_count -= 1
            if true_pred_overlap_count != 1: # Only accept preds that overlap one label
                continue
            true_id = -1
            for possible_id in np.unique(pred_activated_true):
                if possible_id == 0:
                    continue
                true_id = possible_id
                break
            true_instance = np.where(true == true_id, 1, 0)
            iou = self.calc_simple_iou_score(true_instance, pred_instance)
            if iou > min_iou:
                valid_preds += pred_instance
        return valid_preds
    def calc_f1_score(self, precision, recall):
        f1_score = (2*precision*recall)/(precision+recall)
        return f1_score

    def calc_cummulative_threshold_iou(self, true_segs, pred_segs):
        iou_scores = []
        for region_alias in true_segs.keys():
            true_seg = true_segs[region_alias].copy()
            pred_seg = pred_segs[region_alias].copy()
            # CSV CODE
            activated_true = np.where(true_seg > 0, 1, 0)
            activated_pred = np.where(pred_seg > 0, 1, 0)

            # IOU
            iou_score = self.calc_simple_iou_score(activated_true, activated_pred)
            iou_scores.append(iou_score)
        return np.mean(iou_scores)
    def calc_precision(self, true, pred):
        true_activation = np.where(true > 0, 1, 0)
        true_activated_pred = pred*true_activation
        label_count = len(np.unique(true_activated_pred))
        if 0 in np.unique(true_activated_pred):
            label_count -= 1
        total_pred = len(np.unique(pred))
        if 0 in np.unique(pred):
            total_pred -= 1
        # print("p lab:", label_count)
        # print("p pred:", total_pred)
        precision = -1
        if total_pred != 0:
            precision = label_count/total_pred
        return precision
    def calc_recall(self, true, pred):
        true_activation = np.where(true > 0, 1, 0)
        true_activated_pred = pred*true_activation
        label_count = len(np.unique(true_activated_pred))
        if 0 in np.unique(true_activated_pred):
            label_count -= 1
        total_true = len(np.unique(true))
        if 0 in np.unique(true):
            total_true -= 1
        # print("r lab:", label_count)
        # print("r true:", total_true)
        recall = 0
        if total_true != 0:
            recall = label_count/total_true
        return recall

    def calc_tpr(self, true, pred):
        tpr = self.calc_recall(true, pred)
        return tpr
    def calc_fpr(self, true, pred):
        pass
    def calc_precision_recall_auc(self, true, threshold_sweep):
        # NOT COMPLETE YET
        if len(threshold_sweep) < 2:
            print("Not enough treshold output. Need 2 or more") #Make exception later
        precisions = []
        recalls = []
        for pred in threshold_sweep:
            precision = self.calc_precision(true, pred)
            if precision != -1:
                recall = self.calc_recall(true, pred)
                precisions.append(precision)
                recalls.append(recall)
        auc = self.calc_auc(recalls, precisions)
        return auc, precisions, recalls

    def calc_auc(self, x_vals, y_vals):
        # This is an imprecise method
        auc = 0
        for i in range(len(y_vals)-1):
            curr_y = y_vals[i]
            curr_x = x_vals[i]
            next_y = y_vals[i+1]
            next_x = x_vals[i+1]
            width = abs(curr_x - next_x)
            height = abs(next_y - curr_y)
            sub_area = width*height
            auc += sub_area
        return auc

    def calc_simple_iou_score(self, true, pred):
        intersection = true + pred
        intersection = np.where(intersection == 2, 1, 0)
        intersection_score = np.sum(intersection)

        union = true + pred
        union = np.where(union > 0, 1, 0)
        union_score = np.sum(union)

        iou = intersection_score/union_score

        return iou
