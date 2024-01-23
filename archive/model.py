import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def main():
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=1, 
                                              shuffle=True, 
                                              num_workers=4,
                                              collate_fn=utils.collate_fn
                                             ) 

    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=1, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn
                                                  ) 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005) 

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.1) 
    num_epochs = 150

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
    
    # pick one image from the test set
    img, _ = dataset_test[1]
    # put the model in evaluation mode
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

class EuropaIceBlockDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, train=0):
        self.root = root
        self.transforms = transforms
        self.train = train
        if self.train==0:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "ImageSeg/train"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "MaskSeg/train"))))
        elif self.train==1:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "ImageSeg/val"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "MaskSeg/val"))))
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "ImageSeg/test"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "MaskSeg/test"))))
    def __getitem__(self, idx):
        if self.train==0:
            img_path = os.path.join(self.root, "ImageSeg/train", self.imgs[idx])
            mask_path = os.path.join(self.root, "MaskSeg/train", self.masks[idx])
        elif self.train==1:
            img_path = os.path.join(self.root, "ImageSeg/val", self.imgs[idx])
            mask_path = os.path.join(self.root, "MaskSeg/val", self.masks[idx])
        else:
            img_path = os.path.join(self.root, "ImageSeg/test", self.imgs[idx])
            mask_path = os.path.join(self.root, "MaskSeg/test", self.masks[idx])
        img = Image.open(img_path).convert("RGB") 
        img_array = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)
        mask = np.array(mask)
       
        # instances are encoded as different colors  
        obj_ids = np.unique(mask)
        # first id is the background, so remove it 
        obj_ids =  obj_ids[1:]
        # split the color-encoded mask into a set  of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        
        for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          if xmin<xmax and ymin<ymax:
            boxes.append([xmin, ymin, xmax, ymax])
        num_boxes = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_boxes,), dtype=torch.int64) 
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        try: 
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Error thrown here by number of images: https://discuss.pytorch.org/t/how-to-solve-indexerror-too-many-indices-for-tensor-of-dimension-1/40168/3
        except IndexError:
            area = (0)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, 
                                                               box_score_thresh=0.75, 
                                                               trainable_backbone_layers=3)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    main()