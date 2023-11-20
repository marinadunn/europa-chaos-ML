import src.config as config

import os
import random
from glob import glob
from typing import List, Optional

# arrays
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches

# image processing
import cv2
from PIL import Image

# scikit-learn
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# torch
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid

# Fixing global random seed for reproducibility
np.random.seed(42)
random.seed(42)


def show_images(img_path: str,
                title: Optional[str] = None,
                cmap: Optional[str] = None
                ) -> None:
    """
    Displays original chaos region image

    Parameters:
        img_path (str):
            Filepath where original image file is located.
        title (str):
            Optional title to use as the plot title.
        cmap (str):
            Optional built-in matplotlib colormap instance to use as the
                setting when plotting.
    """

    img = cv2.imread(img_path)

    plt.figure(figsize = (8, 8))
    plt.imshow(img, cmap)
    plt.title(title, fontsize = 18)
    plt.axis('off')
    plt.show()


def show_masks(mask_path: str,
               title: Optional[str] = None,
               cmap: Optional[str] = None
               ) -> List[str]:
    """
    Displays chaos label masks

    Parameters:
        mask_path (str):
            File directory where cropped ground truth PNG masks are located.
        title (str):
            Optional title to use as the plot title.
        cmap (str):
            Optional built-in matplotlib colormap instance to use as the
                setting when plotting.
    """

    # glob all mask PNG images together
    masks = sorted(glob(f'{mask_path}/*.png'))

    # each mask instance has a unique color
    fig, axs = plt.subplots(5, 5, sharex = True, sharey = True, figsize = (10, 10))

    # loop through masks and axes
    for mask, ax in zip(masks, axs.ravel()):
        m = plt.imread(mask)
        ax.imshow(m, cmap)
        ax.set_title(title, fontsize = 18)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return masks


def show_dataset(img_dir: str, lbl_dir: str) -> None:
    """
    Displays original image tile in the first column, the ground truth
    mask tile in the second column, and the mask overlayed on the image
    in the third column for each image and mask in a dataset.

    Parameters:
        img_dir (str):
            File directory where cropped PNG image tiles are located.
        lbl_dir (str):
            File directory where cropped PNG ground truth mask PNG tiles
            are located.
    """

    for file in os.listdir(img_dir):
        if file.endswith(".png"):

            # Load the image and mask
            img = cv2.imread(os.path.join(img_dir, file), 0)
            lbl = cv2.imread(os.path.join(lbl_dir, file))

            name = os.path.splitext(os.path.basename(file))[0]
            print('Processing: ', name)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))

            ax1.imshow(img, cmap='gray')
            ax1.set_title('Original Chaos Image')
            ax1.axis('off')

            ax2.imshow(lbl)
            ax2.set_title("Human-Labeled Plates")
            ax2.axis('off')

            ax3.imshow(img, cmap='gray')
            ax3.imshow(lbl, alpha = 0.5)
            ax3.set_title("Plate Labels Overlayed on Chaos")
            ax3.axis('off')

            plt.tight_layout()
            plt.show()


def plot_loss(loss_list, num_epochs: int) -> None:
    """
    Plots the loss training history for a PyTorch model.

    Parameters:
        loss_list (List[float]):
            A list of loss values obtained during training.
        num_epochs (int):
            Total number of epochs model was trained for.
    """

    epochs = np.arange(num_epochs, dtype=int)

    plt.figure(figsize=(6,6))
    plt.tight_layout()
    plt.plot(epochs, loss_list)
    plt.title("Training Loss History", fontsize=22)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.savefig(f'{config.OPTUNA_OUTPUT_PATH}/loss_history.png',
                bbox_inches='tight', dpi=300)


def plot_precision_recall(precisions, recalls, AP):
    """
    Draw precision-recall curve.

    AP (float):
        Average precision at IoU >= 0.5
    precisions (List[float]):
        List of precision values
    recalls (List[float]):
        List of recall values
    """
    # Plot the Precision-Recall curve
    plt.figure(figsize=(6, 6))
    disp = PrecisionRecallDisplay(precision=precisions,
                                  recall=recalls,
                                  average_precision=AP)
    disp.plot()
    plt.title("Precision-Recall Curve. AP@50 = {:.3f}".format(np.round(AP, 3)))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_OUTPUT_PATH}/precision-recall.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_ROC_curve(TP, TN, FP, FN):
    """
    Draw ROC curve.

    Parameters:
        TP (int):
            True positive predictions count.
        TN (int):
            True negative predictions count.
        FP (int):
            False positive predictions count.
        FN (int):
            False negative predictions count.
    """

    fpr = FP / (FP + TN)  # false positive rate
    tpr = TP / (TP + FN)  # true positive rate
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
    disp.plot()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_OUTPUT_PATH}/ROC_curve.png',
                bbox_inches='tight', dpi=300)
    plt.show()


def visualize_predictions(image,
                          pred_masks,
                          pred_boxes,
                          pred_class_ids,
                          pred_scores,
                          iou,
                          accuracy,
                          precision,
                          recall,
                          iou_threshold
                          ):
    """
    Visualizes predictions from a Mask R-CNN model for a given image and
    predictions using NumPy arrays and matplotlib.

    Args:
        image (np.ndarray):
            Original input image of shape (channels, height, width).
        pred_boxes (np.ndarray):
            Predicted bounding boxes (num_instances, 4).
        pred_class_ids (np.ndarray):
            Predicted class IDs (num_instances,).
        pred_masks (np.ndarray):
            Predicted masks (num_instances, height, width).
        pred_scores (np.ndarray):
            Predicted scores/confidences (num_instances,).
        iou (float):
            Segmentation mask IoU score.
        accuracy (float):
            Accuracy score.
        precision (float):
            Precision score.
        recall (float):
            Recall score.
        iou_threshold (float):
            IoU threshold used to calculate the scores.
    """

    # create figure and axes
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 6))

    # convert image to numpy array
    image = image.permute(1, 2, 0)
    ax.imshow(image, cmap='gray')

    class_names = ['Background', 'Plate']

    # Iterate through predictions
    for box, class_id, mask, score in zip(pred_boxes,
                                          pred_class_ids,
                                          pred_masks,
                                          pred_scores):
        x_min, y_min, x_max, y_max = box
        class_name = class_names[class_id]

        # Draw bounding box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw mask
        mask_binary = (mask > 0.5).astype(np.uint8)  # Threshold mask, convert to binary
        mask_color = np.random.rand(3)
        ax.imshow(np.dstack((mask_color * mask_binary,
                             mask_binary * 0,
                             mask_binary * 0,
                             mask_binary * 0.5)
                            ),
                  interpolation='none',
                  alpha=0.4)

        # Add class name and score
        label = f"{class_name}: {score:.2f}"
        ax.text(x_min, y_min, label, color='w', backgroundcolor='r')

    # Set axis properties
    fig.suptitle("Mask R-CNN Predictions Overlayed on Image", fontsize=18)
    ax.axis('off')
    ax.set_title(f"Predictions:\n"
                 f"IoU: {iou:.2f}, Accuracy: {accuracy:.2f}, "
                 f"Precision: {precision:.2f}, Recall: {recall:.2f}\n"
                 f"IoU Threshold: {iou_threshold:.2f}")
    plt.tight_layout()
    plt.show()


def create_confusion_matrix(TP, TN, FP, FN):
    """
    Creates a confusion matrix given true positive (TP),
    true negative (TN), false positive (FP), and false
    negative (FN) counts.

    Parameters:
        TP (int):
            True positive predictions count.
        TN (int):
            True negative predictions count.
        FP (int):
            False positive predictions count.
        FN (int):
            False negative predictions count.

    Returns:
        cm (np.ndarray):
            Confusion matrix as a 2x2 numpy array.
    """

    cm = np.array([[TP, FP], [FN, TN]])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Background', 'Plate'])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax)
    plt.savefig(f'{config.OPTUNA_OUTPUT_PATH}/confusion_matrix.png',
                bbox_inches='tight', dpi=300)
