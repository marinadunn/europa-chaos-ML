import os
import numpy as np
import torch
import random
import cv2

# Use directly in model training
class EuropaIceBlockDataset(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset from the torch.utils.data.Dataset class,
    applies optional transforms.
    """

    def __init__(self, root, img_dir, lbl_dir, config, transforms=None):
        """
        Initializes variables.

        Parameters:
            root (str):
                Root directory for data.
            img_dir (str):
                Directory for images.
            lbl_dir (str):
                Directory for ground truth mask PNGs.
            transforms (Optional[torchvision.transforms.Compose]):
                Optional transforms to apply to images and masks.
        """
        self.root = root
        self.transforms = transforms
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.config = config
        img_dir_files = sorted(os.listdir(os.path.join(root, img_dir)))
        lbl_dir_files = sorted(os.listdir(os.path.join(root, lbl_dir)))
        self.imgs = self.get_npy_files(img_dir_files)
        self.masks = self.get_npy_files(lbl_dir_files)

    def get_npy_files(self, files):
        return [file for file in files if file.endswith(".npy")]

    def get_crop_metadata(self, file, img_shape):
        """
        Extracts crop metadata from the given file name and image shape.

        Parameters:
        - file (str): The name of the file containing crop information.
        - img_shape (tuple): The shape of the original image in the format (height, width).

        Returns:
        tuple: A tuple containing region alias and a tuple representing the crop coordinates (y_min, x_min, y_max, x_max).
        """
        # Extracting region alias and coordinates from the file name
        file_name = os.path.splitext(file)[0]
        split_file = file_name.split('_')
        region_alias = split_file[0]
        x = int(split_file[1])
        y = int(split_file[2])

        # Extracting image dimensions
        h = img_shape[0]
        w = img_shape[1]

        # Calculating crop coordinates
        y_min = y
        y_max = y_min + h
        x_min = x
        x_max = x_min + w

        return region_alias, (y_min, x_min, y_max, x_max)

    def __getitem__(self, idx):
        """
        Forms a PyTorch dataset. Inherits lists of filepaths for images and
        labels, and optional transforms. For each image and label, opens them
        with PIL, converts to numpy array. Unique objects instances are detected,
        and masks are split into binary masks. Bounding boxes are then calculated
        for each object instance in the mask, and box areas are calculated.
        Finally, each image and label are converted to PyTorch tensors, and
        optional transforms are applied if specified.

        Parameters
            idx (int):
                Index of image and label to be processed.

        Returns:
            img (torch.Tensor):
                PyTorch tensor of image with shape [channels, H, W]

            target (Dict[str, torch.Tensor]):
                A dict containing the following fields:
                    - boxes (FloatTensor[N, 4]): the coordinates of the N
                        bounding boxes of [x0, y0, x1, y1] format, ranging
                        from 0 to W and 0 to H
                    - labels (Int64Tensor[N]): the label for each bounding box
                    - image_id (Int64Tensor[1]): an image identifier. Should
                        be unique between all images in dataset, and is used
                        during evaluation.
                    - area (Tensor[N]): The area of the bounding box.
                    - iscrowd (UInt8Tensor[N]): instances with iscrowd=True
                        will be ignored during evaluation.
                    - masks (UInt8Tensor[N, H, W]): The segmentation masks
                        for each one of the objects.

            crop_metadata (Tuple[str, List[int]]):
                A tuple containing the following fields:
                    - region_alias (str): the alias of the region
                    - [x, y, w, h] (List[int]): the coordinates of the
                        bounding box of the region
        """

        # open each image, mask numpy file
        img = np.load(os.path.join(self.root, self.img_dir, self.imgs[idx]), allow_pickle=True)
        mask = np.load(os.path.join(self.root, self.lbl_dir, self.masks[idx]), allow_pickle=True)
        crop_metadata = self.get_crop_metadata(self.imgs[idx], img.shape)

        # instances are encoded as different colors, with 0 being background
        obj_ids = np.unique(mask)[1:]  # exclude background

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]  # shape is [num_instances, H, W]

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i].nonzero())
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])

            # coords of bounding boxes in [x1, y1, x2, y2] format
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        if len(boxes) > 0:
            # shape is [num_instances, [x1, y1, x2, y2]]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # label for each ground-truth bounding box; there is only one class
        labels = torch.ones(len(obj_ids), dtype=torch.int64)

        # segmentation binary masks for each instance [num_instances, H, W]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # unique integer image identifier, used during evaluation
        image_id = torch.tensor([idx])

        # area of the bounding box: (y2 - y1) * (x2 - x1)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Crowd box in COCO is a bounding box around several instances. Thus,
        # suppose all instances are not crowd. Instances with iscrowd=True
        # will be ignored during evaluation
        iscrowd = torch.zeros(len(obj_ids), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, crop_metadata

    def __len__(self):
        return len(self.imgs)
