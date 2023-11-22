import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

import config as config

class AbstractMaskRCNN():
    """
    Abstract class that provides methods relevant for rcnn and trainable_rcnn.
    Not designed for direct use.
    """
    def __init__(self):
        """Initialize the AbstractMaskRCNN class."""
        self.device = config.device
        self.model = None
        self.train_size = 1024

    def get_model(self):
        """Get the current model."""
        return self.model

    def set_model(self, model):
        """Set the model and move it to the specified device."""
        self.model = model.to(self.device)

    def load_and_set_model(self,
                           arch_id,
                           num_classes,
                           trainable_layers=3,
                           type="v1",
                           dropout_factor=0.1,
                           model_path=""
                           ):
        """
        Load a pre-trained model, modify its architecture, and set it as the current model.

        Args:
            arch_id (int): Identifier for the model architecture.
            num_classes (int): Number of output classes.
            trainable_layers (int): Number of trainable layers in the backbone.
            type (str): Type of Mask R-CNN model architecture version ("v1" or "v2").
            dropout_factor (float): Dropout factor for the mask predictor head.
            model_path (str): Path to a pre-trained model file.
        """
        model = self.define_model(arch_id, num_classes, trainable_layers, type, dropout_factor)
        if model_path != "":
            model.load_state_dict(torch.load(model_path))
        self.set_model(model)

    def create_mask_predictor_head(self,
                                   in_features_mask,
                                   dropout_factor,
                                   hidden_layer,
                                   num_classes
                                   ):
        """
        Create a new mask predictor head for the model.

        Args:
            in_features_mask (int): Number of input features for the mask predictor head.
            dropout_factor (float): Dropout factor for the mask predictor head.
            hidden_layer (int): Number of hidden units in the mask predictor head.
            num_classes (int): Number of output classes.

        Returns:
            nn.Sequential: New mask predictor head.
        """
        layers = []
        layer_sizes = [in_features_mask, int(in_features_mask/2), int(in_features_mask/4),
                        int(in_features_mask/8), int(in_features_mask/16)]

        for layer_size in layer_sizes:
            layers.append(nn.Conv2d(layer_size, layer_size//2, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1), bias=False))
            layers.append(nn.ReLU())
            if dropout_factor > 0:
                layers.append(nn.Dropout(dropout_factor))

        layers.append(MaskRCNNPredictor(layer_sizes[-1], hidden_layer, num_classes))
        return nn.Sequential(*layers)

    def define_model(self,
                     arch_id,
                     num_classes,
                     trainable_layers=3,
                     type="v1",
                     dropout_factor=0.1):
        """
        Define the model architecture, replace its box and mask predictor heads, and return the model.

        Args:
            arch_id (int): Identifier for the model architecture.
            num_classes (int): Number of output classes.
            trainable_layers (int): Number of trainable layers in the backbone.
            type (str): Type of model architecture ("v1" or "v2").
            dropout_factor (float): Dropout factor for the mask predictor head.

        Returns:
            torch.nn.Module: Defined model.
        """
        model = None

        if arch_id == 1:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                trainable_backbone_layers=trainable_layers)
        elif arch_id == 2:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                pretrained=True,
                trainable_backbone_layers=trainable_layers)

       # Replace box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        new_head = self.create_mask_predictor_head(in_features_mask, dropout_factor, hidden_layer, num_classes)
        model.roi_heads.mask_predictor = new_head

        return model
