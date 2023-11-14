
import os
import numpy as np
import torch
import random
import cv2

class EuropaIceBlockMetaset(torch.utils.data.Dataset):
    # Use in analysis after model building, provides metadata on each observation
    def __init__(self, dataset_type, root, img_dir, lbl_dir, transforms=None):
        self.dataset_type = dataset_type
        self.root = root
        self.transforms = transforms
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        img_dir_files = list(sorted(os.listdir(os.path.join(root, img_dir))))
        lbl_dir_files = list(sorted(os.listdir(os.path.join(root, lbl_dir))))
        self.imgs = self.get_npy_files(img_dir_files)
        self.masks = self.get_npy_files(lbl_dir_files)
    def get_npy_files(self, files):
        npy_list = []
        for file in files:
            split_file = file.split(".")
            ext = split_file[-1]
            if ext == "npy":
                npy_list.append(file)
        return npy_list
    def get_crop_metadata(self, file, img_shape):
        file_name = file.split('.')[0]
        split_file = file_name.split('_')
        alias = split_file[0]
        x = int(split_file[1])
        y = int(split_file[2])
        h = img_shape[0]
        w = img_shape[1]
        return (alias, [x, y, w, h])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.root, self.img_dir, self.imgs[idx]))
        mask = np.load(os.path.join(self.root, self.lbl_dir, self.masks[idx]))
        crop_metadata = self.get_crop_metadata(self.imgs[idx], img.shape)
        obj_ids = np.unique(mask)[1:] 
        masks = (mask == obj_ids[:, None,None]) # This code always confused me with the syntax
        boxes = []
        for i in range(len(obj_ids)): # I think Ahmed wanted better logic here?
          pos = np.where(masks[i]) 
          boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])   
        #come back to this with tensor versus numpy array
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = dict([("boxes", boxes),
                       ("area", area),
                       ("labels", torch.ones((len(obj_ids),), dtype=torch.int64) ),
                       ("masks", torch.as_tensor(masks, dtype=torch.uint8)), 
                       ("image_id", torch.tensor([idx])), 
                       ("iscrowd", torch.zeros((len(obj_ids),), dtype=torch.int64))])
        #import pdb; pdb.set_trace()
        if self.transforms is not None: # double check this??
            img, target = self.transforms(img, target)
        #import pdb; pdb.set_trace()
        return img, target, crop_metadata

class EuropaIceBlockDataset(torch.utils.data.Dataset):
    def __init__(self,root, img_dir, lbl_dir, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        img_dir_files = list(sorted(os.listdir(os.path.join(root, img_dir))))
        lbl_dir_files = list(sorted(os.listdir(os.path.join(root, lbl_dir))))
        self.imgs = self.get_npy_files(img_dir_files)
        self.masks = self.get_npy_files(lbl_dir_files)
    def get_npy_files(self, files):
        npy_list = []
        for file in files:
            split_file = file.split(".")
            ext = split_file[-1]
            if ext == "npy":
                npy_list.append(file)
        return npy_list
    def get_crop_metadata(self, file, img_shape):
        file_name = file.split('.')[0]
        split_file = file_name.split('_')
        alias = split_file[0]
        x = int(split_file[1])
        y = int(split_file[2])
        h = img_shape[0]
        w = img_shape[1]
        return (alias, [x, y, w, h])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.root, self.img_dir, self.imgs[idx]))
        mask = np.load(os.path.join(self.root, self.lbl_dir, self.masks[idx]))
        crop_metadata = self.get_crop_metadata(self.imgs[idx], img.shape)
        obj_ids = np.unique(mask)[1:] 
        masks = (mask == obj_ids[:, None,None]) # This code always confused me with the syntax
        boxes = []
        for i in range(len(obj_ids)): # I think Ahmed wanted better logic here?
          pos = np.where(masks[i]) 
          boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])   
        #come back to this with tensor versus numpy array
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = dict([("boxes", boxes),
                       ("area", area),
                       ("labels", torch.ones((len(obj_ids),), dtype=torch.int64) ),
                       ("masks", torch.as_tensor(masks, dtype=torch.uint8)), 
                       ("image_id", torch.tensor([idx])), 
                       ("iscrowd", torch.zeros((len(obj_ids),), dtype=torch.int64))])
        #import pdb; pdb.set_trace()
        if self.transforms is not None: # double check this??
            img, target = self.transforms(img, target)
        #import pdb; pdb.set_trace()
        return img, target

class EuropaIceBlockDatasetTorch(torch.utils.data.Dataset):
    # Use directly in model training
    def __init__(self, root, img_dir, lbl_dir, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        img_dir_files = list(sorted(os.listdir(os.path.join(root, img_dir))))
        lbl_dir_files = list(sorted(os.listdir(os.path.join(root, lbl_dir))))
        self.imgs = self.get_npy_files(img_dir_files)
        self.masks = self.get_npy_files(lbl_dir_files)
    def get_npy_files(self, files):
        npy_list = []
        for file in files:
            split_file = file.split(".")
            ext = split_file[-1]
            if ext == "npy":
                npy_list.append(file)
        return npy_list
    def get_crop_metadata(self, file, img_shape):
        file_name = file.split('.')[0]
        split_file = file_name.split('_')
        alias = split_file[0]
        x = int(split_file[1])
        y = int(split_file[2])
        h = img_shape[0]
        w = img_shape[1]
        return (alias, [x, y, w, h])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.root, self.img_dir, self.imgs[idx]))
        mask = np.load(os.path.join(self.root, self.lbl_dir, self.masks[idx]))
        crop_metadata = self.get_crop_metadata(self.imgs[idx], img.shape)
        obj_ids = np.unique(mask)[1:] 
        masks_og = (mask == obj_ids[:, None,None]) # This code always confused me with the syntax
        #come back to this with tensor versus numpy array

        boxes = []
        while len(boxes) == 0:
            # import pdb; pdb.set_trace()
            seed = np.random.randint(2147482647)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.transforms is not None: # double check this??
                transformed_img = self.transforms(img)

            random.seed(seed)
            torch.manual_seed(seed)
            if self.transforms is not None: # double check this??
                transformed_masks = self.transforms(masks_og)


            np_masks = transformed_masks.squeeze().numpy().swapaxes(0,1)
            for i in range(len(obj_ids)): # I think Ahmed wanted better logic here?
                pos = np.where(np_masks[i]) 
                if (pos[0].shape[0] == 0) or (pos[1].shape[0] == 0): # check if valid label even exists
                    continue
                box = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
                box_area = (box[2] - box[0])*(box[3] - box[1])
                if box_area > 50:
                    boxes.append(box) #xmin, ymin, xmax, ymax  

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = dict([("boxes", boxes),
                       ("area", area),
                       ("labels", torch.ones((len(obj_ids),), dtype=torch.int64) ),
                       ("masks", transformed_masks), 
                       ("image_id", torch.tensor([idx])), 
                       ("iscrowd", torch.zeros((len(obj_ids),), dtype=torch.int64))])


        
        """ If youre using the notebook code the below code is how u transform"""
        # if self.transforms is not None: # double check this??
        #     img, target = self.transforms(img, target)
        return transformed_img, target