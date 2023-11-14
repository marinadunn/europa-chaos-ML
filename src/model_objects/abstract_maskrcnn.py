import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

class AbstractMaskRCNN():
    # Abstract class that provides methods relevant for rcnn and trainable_rcnn
    # Not designed for direct use
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        self.train_size = 1024

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def load_and_set_model(self, arch_id, num_classes,  trainable_layers=3, type="v1", dropout_factor=0.1, model_path=""):
        model = None
        if arch_id == 1:
            model = self.define_model_01(num_classes, trainable_layers=trainable_layers, type=type)
        elif arch_id == 2:
            model = self.define_model_02(num_classes, trainable_layers=trainable_layers, type=type, dropout_factor=dropout_factor)
        if model_path != "":
            model.load_state_dict(torch.load(model_path))
        self.set_model(model)

    # def load_and_set_model(self, arch_id, num_classes, model_path=""):
    #     model = None
    #     if arch_id == 1:
    #         model = self.define_model_01(num_classes)
    #     if model_path != "":
    #         model.load_state_dict(torch.load(model_path))
    #     self.set_model(model)

    def define_model_01(self, num_classes, trainable_layers=3, type="v1"):
        model = None
        if type == "v1":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained = True,
                trainable_backbone_layers=trainable_layers)
        elif type == "v2":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                pretrained = True,
                trainable_backbone_layers=trainable_layers)

        # Replace box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        new_head = nn.Sequential(
                nn.Conv2d(in_features_mask, int(in_features_mask/2), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Conv2d(int(in_features_mask/2), int(in_features_mask/4), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Conv2d(int(in_features_mask/4), int(in_features_mask/8), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Conv2d(int(in_features_mask/8), int(in_features_mask/16), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                MaskRCNNPredictor(int(in_features_mask/16), hidden_layer, num_classes))
        model.roi_heads.mask_predictor = new_head
        return model

    def define_model_02(self, num_classes, trainable_layers=3, type="v1", dropout_factor=0.1):
        model = None
        if type == "v1":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained = True,
                trainable_backbone_layers=trainable_layers)
        elif type == "v2":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                pretrained = True,
                trainable_backbone_layers=trainable_layers)

        # Replace box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        new_head = nn.Sequential(
                nn.Conv2d(in_features_mask, int(in_features_mask/2), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Dropout(dropout_factor),
                nn.Conv2d(int(in_features_mask/2), int(in_features_mask/4), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Dropout(dropout_factor),
                nn.Conv2d(int(in_features_mask/4), int(in_features_mask/8), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Dropout(dropout_factor),
                nn.Conv2d(int(in_features_mask/8), int(in_features_mask/16), kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                nn.ReLU(),
                nn.Dropout(dropout_factor),
                MaskRCNNPredictor(int(in_features_mask/16), hidden_layer, num_classes))
        model.roi_heads.mask_predictor = new_head
        return model
