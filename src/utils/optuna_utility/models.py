import optuna
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2

# Function designed such that optuna can interact with a MaskrCNN in a study
# Might make a class for this down the road

def define_mrcnn_model_01(trial, num_classes):
    """
    Define Mask R-CNN model with architecture modifications for hyperparameter tuning.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
        num_classes (int): Number of output classes.

    Returns:
        torchvision.models.detection.maskrcnn_resnet50_fpn: Modified Mask R-CNN model.
    """
    # load Mask R-CNN model with ResNet50 backbone
    model_type = trial.suggest_categorical('model_type', ["v1", "v2"])
    # model = maskrcnn_resnet50_fpn(pretrained = True)

    # Define Mask R-CNN model version
    if model_type == "v1":
        model = maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone)
    elif model_type == "v2":
        model = maskrcnn_resnet50_fpn_v2(pretrained=True, trainable_backbone_layers=trainable_backbone)

    # Get number of input features for bbox classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace pre-trained box predictor head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get number of input features for mask classifier
    hidden_layer = 256

    new_head = nn.Sequential(
        nn.Conv2d(in_features_mask, int(in_features_mask/2), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Conv2d(int(in_features_mask/2), int(in_features_mask/4), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Conv2d(int(in_features_mask/4), int(in_features_mask/8), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Conv2d(int(in_features_mask/8), int(in_features_mask/16), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        MaskRCNNPredictor(int(in_features_mask/16), hidden_layer, num_classes)
    )

    model.roi_heads.mask_predictor = new_head

    return model

def define_mrcnn_model_02(trial, num_classes):
    """
    Define Mask R-CNN model with hyperparameter tuning using Optuna.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
        num_classes (int): Number of output classes.

    Returns:
        torchvision.models.detection.maskrcnn_resnet50_fpn or
            torchvision.models.detection.maskrcnn_resnet50_fpn_v2: Modified Mask R-CNN model.
    """
    # Define optuna hyperparameters
    model_type = trial.suggest_categorical('model_type', ["v1", "v2"])
    trainable_backbone = trial.suggest_categorical("trainable_backbone_layers", [0, 3, 5])
    dropout_factor = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    model = None

    # Define Mask R-CNN model version
    if model_type == "v1":
        model = maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone)
    elif model_type == "v2":
        model = maskrcnn_resnet50_fpn_v2(pretrained=True, trainable_backbone_layers=trainable_backbone)

    # Get number of input features for bbox classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace pre-trained box predictor head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get number of input features for mask classifier
    hidden_layer = 256

    new_head = nn.Sequential(
        nn.Conv2d(in_features_mask, int(in_features_mask/2), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Dropout(dropout_factor),
        nn.Conv2d(int(in_features_mask/2), int(in_features_mask/4), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Dropout(dropout_factor),
        nn.Conv2d(int(in_features_mask/4), int(in_features_mask/8), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Dropout(dropout_factor),
        nn.Conv2d(int(in_features_mask/8), int(in_features_mask/16), kernel_size=(3, 3),
                  stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Dropout(dropout_factor),
        MaskRCNNPredictor(int(in_features_mask/16), hidden_layer, num_classes)
    )

    model.roi_heads.mask_predictor = new_head

    return model
