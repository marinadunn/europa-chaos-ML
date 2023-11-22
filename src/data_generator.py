import os
import numpy as np
from PIL import Image
import cv2
from utils.file_utils import clear_and_remake_directory
from config import (IMG_TEST_PATH, IMG_TRAIN_PATH,
                    LBL_TEST_PATH, LBL_TRAIN_PATH,
                    PROCESSED_DATA_PATH,
                    CHAOS_REGION_ALIAS_TO_FILE_MAP,
                    CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                    CHAOS_REGION_ALIAS_TO_REGION_MAP
                    )


class DataGenerator():
    """
    Generates data based on the Europa dataset structure.
    """
    def __init__(self):
        """
        Initialize the DataGenerator class.
        """
        # Obtain path names from the config file
        self.processed_path = PROCESSED_DATA_PATH
        self.train_img_path = IMG_TRAIN_PATH
        self.train_lbl_path = LBL_TRAIN_PATH
        self.test_img_path = IMG_TEST_PATH
        self.test_lbl_path = LBL_TEST_PATH
        self.prev_settings = None
        self.prev_function = None

    def reset_processed_data_dir(self):
        """
        Clear and remake the processed data directory, subdirectories recursively.
        """
        clear_and_remake_directory(self.processed_path)
        os.makedirs(self.train_img_path)
        os.makedirs(self.train_lbl_path)
        os.makedirs(self.test_img_path)
        os.makedirs(self.test_lbl_path)

    def same_method_and_settings(self, curr_function, curr_settings):
        """
        Check if the current function and settings are the same as the previous.
        """
        if self.prev_settings == curr_settings and self.prev_function == curr_function:
            return True
        else:
            return False

    def sliding_crops_experiment(self,
                                 train_regions,
                                 test_regions,
                                 crop_size,
                                 stride,
                                 min_sheet_area
                                 ):
        """
        Generate sliding crops of specified size and stride without pruning labels.

        Args:
            train_regions (list): List of regions for training.
            test_regions (list): List of regions for testing.
            crop_size (int): Uniform height/width of the cropped images.
            stride (int): Stride for sliding window.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        curr_settings = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_size": crop_size,
            "stride": stride,
            "min_sheet_area": min_sheet_area
        }

        curr_function = "sliding_crops_experiment"
        if self.same_method_and_settings(curr_function, curr_settings):
            return
        else:
            self.prev_settings = curr_settings
            self.prev_function = curr_function

        print("Generating Data...")
        self.reset_processed_data_dir()

        for train_region in train_regions:
            self.gen_sliding_crops(
                "train",
                train_region,
                crop_size,
                stride,
                min_sheet_area)

        for test_region in test_regions:
            self.gen_sliding_crops(
                "test",
                test_region,
                crop_size,
                stride,
                min_sheet_area)

        print("Finished!")

    def pruned_sliding_crops_experiment(self,
                                        train_regions,
                                        test_regions,
                                        crop_size,
                                        stride,
                                        min_sheet_area
                                        ):
        """
        Generate sliding crops of specified size and stride, prune labels that
        are cut off, regardless of size.

        Args:
            train_regions (list): List of regions for training.
            test_regions (list): List of regions for testing.
            crop_size (int): Uniform height/width of the cropped images.
            stride (int): Stride for sliding window.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        curr_settings = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_size": crop_size,
            "stride": stride,
            "min_sheet_area": min_sheet_area
        }

        curr_function = "pruned_sliding_crops_experiment"
        if self.same_method_and_settings(curr_function, curr_settings):
            return
        else:
            self.prev_settings = curr_settings
            self.prev_function = curr_function

        print("Generating Data...")
        self.reset_processed_data_dir()

        for train_region in train_regions:
            self.gen_sliding_crops_and_prune_partial_labels(
                "train",
                train_region,
                crop_size,
                stride,
                min_sheet_area
            )

        for test_region in test_regions:
            self.gen_sliding_crops_and_prune_partial_labels(
                "test",
                test_region,
                crop_size,
                stride,
                min_sheet_area
            )

        print("Finished generating data!")

    def get_sweep_crops_and_prune_partial_labels(self,
                                                 region_alias,
                                                 crop_size,
                                                 min_sheet_area
                                                 ):
        """
        Generate sliding crops with pruning for a specific region.

        Args:
            region_alias (str): Alias of the region.
            crop_size (int): Uniform height/width of the cropped images.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        # Load images and labels
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int(((xmax - xmin) - crop_size) + 1)
        max_y_iter = int(((ymax - ymin) - crop_size) + 1)

        # Define starting point
        curr_y_start = ymin - crop_size

        imgs, lbls = [], []

        # Iterate through all possible crops
        for _ in range(max_y_iter):
            curr_y_start += crop_size
            curr_x_start = xmin - crop_size

            for _ in range(max_x_iter):
                curr_x_start += crop_size
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_size)
                lbl_out = self.process_labels(lbl_crop, min_sheet_area)

                # Object instances are encoded as unique colors
                # Ensure there is at least 1 ice block in the image
                if len(np.unique(lbl_out)) > 1:
                    imgs.append(img_crop)
                    lbls.append(lbl_out)

        return imgs, lbls

    def gen_sliding_crops_and_prune_partial_labels(self,
                                                   set_type,
                                                   region_alias,
                                                   crop_size,
                                                   stride,
                                                   min_sheet_area
                                                   ):
        """
        Generate sliding crops of specified size and stride, prune partial labels.

        Args:
            set_type (str): Type of set (train/test).
            region_alias (str): Alias of the region.
            crop_size (int): Uniform height/width of the cropped images.
            stride (int): Stride for sliding window.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        # Get directories for images and labels for the specified set type
        img_dir, lbl_dir = self.get_directories(set_type)

        # Load images and labels
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int((xmax - xmin - crop_size) / stride) + 1
        max_y_iter = int((ymax - ymin - crop_size) / stride) + 1

        # Define starting point
        curr_x_start, curr_y_start = xmin - stride, ymin - stride

        ct = 0  # Initial image

        # Iterate through all possible crops
        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = xmin - stride

            for _ in range(max_x_iter):
                curr_x_start += stride
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_size)
                lbl_out = self.process_labels(lbl_crop, min_sheet_area)

                # Object instances are encoded as unique colors
                # Ensure there is at least 1 ice block in the image
                if len(np.unique(lbl_out)) > 1:
                    # Resize to 1024x1024; should use INTER_CUBIC or INTER_LINEAR when enlarging
                    img_crop = cv2.resize(img_crop, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
                    lbl_out = cv2.resize(lbl_out, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)

                    # Save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # Convert back to images and save
                    self.save_as_images(img_crop, lbl_out, img_dir, lbl_dir, region_alias, x, y, ct)

                    ct += 1  # Move to next image

    def gen_sliding_crops(self,
                          set_type,
                          region_alias,
                          crop_size,
                          stride,
                          min_sheet_area
                          ):
        """
        Generate sliding crops of specified size and stride.

        Args:
            set_type (str): Type of set (train/test).
            region_alias (str): Alias of the region.
            crop_size (int): Uniform height/width of the cropped images.
            stride (int): Stride for sliding window.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        # Get directories for images and labels for the specified set type
        img_dir, lbl_dir = self.get_directories(set_type)

        # Load images and labels
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int((xmax - xmin - crop_size) / stride)
        max_y_iter = int((ymax - ymin - crop_size) / stride)

        # Define starting point
        curr_x_start, curr_y_start = xmin - stride, ymin - stride

        ct = 0  # Initial image

        # Iterate through all possible crops
        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = xmax - stride

            for _ in range(max_x_iter):
                curr_x_start += stride
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_size)
                lbl_out = self.process_labels(lbl_crop, min_sheet_area)

                # Object instances are encoded as unique colors
                # Ensure there is at least 1 ice block in the image
                if len(np.unique(lbl_out)) > 1:
                    # Resize to 1024x1024; should use INTER_CUBIC or INTER_LINEAR when enlarging
                    img_crop = cv2.resize(img_crop, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
                    lbl_out = cv2.resize(lbl_out, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)

                    # Save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # Convert back to images and save
                    self.save_as_images(img_crop, lbl_out, img_dir, lbl_dir, region_alias, x, y, ct)

                    ct += 1  # Move to next image

    def get_directories(self, set_type):
        """
        Get the directories for the specified set type.

        Args:
            set_type (str): Type of set (train/test).
        """
        if set_type == "train":
            return self.train_img_path, self.train_lbl_path
        elif set_type == "test":
            return self.test_img_path, self.test_lbl_path

    def load_images(self, region_alias):
        """
        Load images and labels for the specified region.

        Args:
            region_alias (str): Name of the chaos region.
        """
        # Load images and labels
        img = np.array(Image.open(CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:, :, 0]
        lbl = np.array(Image.open(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:, :, :3]
        print("Images and labels loaded.")

        # Get extent of region
        region_lbl = np.array(Image.open(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias]))
        region_activation = np.where(region_lbl)
        xmin, xmax = np.min(region_activation[1]), np.max(region_activation[1])
        ymin, ymax = np.min(region_activation[0]), np.max(region_activation[0])
        # Get region crop
        region_crop = [xmin, xmax, ymin, ymax]

        return img, lbl, region_crop

    def process_labels(lbl_crop, min_sheet_area):
        """
        Process the labels by pruning small areas and broken labels.

        Args:
            lbl_crop (np.array): Cropped label.
            min_sheet_area (int): Minimum area for a sheet to be considered valid.
        """
        # Initialize processed label
        lbl_crop_processed = np.zeros_like(lbl_crop).astype(int)
        # Convert to structured array
        rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
        struc_data = lbl_crop.view(rgb_type)
        # Get all unique colors
        unique_colors = np.unique(struc_data)
        # Initialize mask id
        mask_id = 1

        # Iterate through all unique colors
        for void_color in unique_colors:
            # Convert to list
            mask_color = void_color.tolist()
            # Skip black
            if mask_color == (0, 0, 0):
                continue

            single_mask = np.all(lbl_crop == mask_color, axis=2)
            # Check if the cropped region is too small or has a broken label
            if DataGenerator.crop_too_small(single_mask, min_sheet_area) or \
                    DataGenerator.crop_has_broken_label(single_mask):
                continue
            # If not, add to the processed label
            else:
                # Add to processed label
                lbl_crop_processed = np.where(single_mask > 0, mask_id, lbl_crop_processed)
                # Increment mask id
                mask_id += 1

        return lbl_crop_processed

    def save_as_images(self,
                       img_crop,
                       lbl_out,
                       img_dir,
                       lbl_dir,
                       region_alias,
                       x, y, ct):
        """
        Convert cropped images and labels back to images and save.

        Args:
            img_crop (np.array): Cropped image.
            lbl_out (np.array): Processed label.
            img_dir (str): Directory for images.
            lbl_dir (str): Directory for labels.
            region_alias (str): Name of the chaos region.
            x (int): x-coordinate of the cropped image.
            y (int): y-coordinate of the cropped image.
            ct (int): Index of the cropped image.
        """
        cv2.imwrite(f'{img_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_crop)
        # Initialize image
        img_png = np.zeros((img_crop.shape[0], img_crop.shape[1], 3))
        # Convert to RGB
        img_png[:, :, 0] = img_crop
        img_png[:, :, 1] = img_crop
        img_png[:, :, 2] = img_crop
        # Add mask to image
        for loc in list(zip(*np.where(lbl_out))):
            img_png[loc[0], loc[1], 0] += 100
        # Save image
        cv2.imwrite(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_png)

    def crop_images(img, lbl, x, y, crop_size):
        """
        Crop the given image and label.

        Args:
            img (np.array): Image.
            lbl (np.array): Label.
            x (int): x-coordinate of the cropped image.
            y (int): y-coordinate of the cropped image.
            crop_size (int): Uniform height/width of the cropped images.
        """
        img_crop = img[y:y + crop_size, x:x + crop_size]
        lbl_crop = lbl[y:y + crop_size, x:x + crop_size]
        return img_crop, lbl_crop

    @staticmethod
    def crop_has_broken_label(lbl_crop):
        """
        Check if the cropped label has a broken label.

        Args:
            lbl_crop (np.array): Cropped label.
        """
        lbl_probe = lbl_crop.copy()
        # Remove the border
        lbl_probe[1:(lbl_probe.shape[0] - 1), 1:(lbl_probe.shape[1] - 1)] = 0
        if len(np.unique(lbl_probe)) > 1:
            return True
        else:
            return False

    @staticmethod
    def crop_too_small(crop, min_area):
        """
        Check if the cropped region is too small.

        Args:
            crop (np.array): Cropped image.
            min_area (int): Minimum area for a sheet to be considered valid.
        """
        # Get the area of the cropped region
        crop_area = np.sum(np.where(crop > 0, 1, 0))
        # Check if the area is too small
        if crop_area < min_area:
            return True
        else:
            return False

