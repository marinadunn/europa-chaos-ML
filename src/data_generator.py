import os
import numpy as np
from PIL import Image
import cv2
import config as config
from utils.file_utils import clear_and_remake_directory


class DataGenerator():
    """
    Generates data based on the Europa dataset structure.
    """
    def __init__(self):
        """
        Initialize the DataGenerator class.
        """
        # Obtain path names from the config file
        self.processed_path = config.PROCESSED_DATA_PATH
        self.train_img_path = config.IMG_TRAIN_PATH
        self.train_lbl_path = config.LBL_TRAIN_PATH
        self.test_img_path = config.IMG_TEST_PATH
        self.test_lbl_path = config.LBL_TEST_PATH
        self.prev_settings = None
        self.prev_function = None
        self.config = config

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
                                 crop_height,
                                 crop_width,
                                 stride,
                                 min_sheet_area
                                 ):
        """
        Generate sliding crops of specified size and stride without pruning labels.
        """
        curr_settings = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_height": crop_height,
            "crop_width": crop_width,
            "stride": stride,
            "min_sheet_area": min_sheet_area
        }
        curr_function = "sliding_crops_experiment"
        return self.prev_settings == curr_settings and self.prev_function == curr_function

        print("Generating Data...")
        self.reset_processed_data_dir()

        for train_region in train_regions:
            self.gen_sliding_crops(
                "train",
                train_region,
                crop_height,
                crop_width,
                stride,
                min_sheet_area)

        for test_region in test_regions:
            self.gen_sliding_crops(
                "test",
                test_region,
                crop_height,
                crop_width,
                stride,
                min_sheet_area)

        print("Finished!")

    def pruned_sliding_crops_experiment(self,
                                        train_regions,
                                        test_regions,
                                        crop_height,
                                        crop_width,
                                        stride,
                                        min_sheet_area
                                        ):
        """
        Generate sliding crops of specified size and stride, prune labels that
        are cut off, regardless of size.
        """
        curr_settings = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_height": crop_height,
            "crop_width": crop_width,
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
                crop_height,
                crop_width,
                stride,
                min_sheet_area
            )

        for test_region in test_regions:
            self.gen_sliding_crops_and_prune_partial_labels(
                "test",
                test_region,
                crop_height,
                crop_width,
                stride,
                min_sheet_area
            )

        print("Finished generating data!")

    def get_sweep_crops_and_prune_partial_labels(self,
                                                 set_type,
                                                 region_alias,
                                                 crop_height,
                                                 crop_width,
                                                 min_sheet_area
                                                 ):
        """
        Generate sliding crops with pruning for a specific region.
        """
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int(((xmax - xmin) - crop_width) + 1)
        max_y_iter = int(((ymax - ymin) - crop_height) + 1)

        curr_y_start = ymin - crop_height

        imgs, lbls = [], []

        for _ in range(max_y_iter):
            curr_y_start += crop_height
            curr_x_start = xmin - crop_width
            for _ in range(max_x_iter):
                curr_x_start += crop_width
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_height, crop_width)
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
                                                   crop_height,
                                                   crop_width,
                                                   stride,
                                                   min_sheet_area
                                                   ):
        """
        Generate sliding crops of specified size and stride, prune partial labels.
        """
        img_dir, lbl_dir = self.get_directories(set_type)
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int((xmax - xmin - crop_width) / stride) + 1
        max_y_iter = int((ymax - ymin - crop_height) / stride) + 1
        curr_x_start, curr_y_start = xmin - stride, ymin - stride

        ct = 0  # Initial image

        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = xmin - stride
            for _ in range(max_x_iter):
                curr_x_start += stride
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_height, crop_width)
                lbl_out = self.process_labels(lbl_crop, min_sheet_area)

                # Object instances are encoded as unique colors
                # Ensure there is at least 1 ice block in the image
                if len(np.unique(lbl_out)) > 1:
                    img_crop, lbl_out = self.resize_images(img_crop, lbl_out, 1024, 1024)

                    # Save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # Convert back to images and save
                    self.save_as_images(img_crop, lbl_out, img_dir, lbl_dir, region_alias, x, y, ct)

                    ct += 1  # Move to next image

    def gen_sliding_crops(self,
                          set_type,
                          region_alias,
                          crop_height,
                          crop_width,
                          stride,
                          min_sheet_area
                          ):
        """
        Generate sliding crops of specified size and stride.
        """
        img_dir, lbl_dir = self.get_directories(set_type)
        img, lbl, region_crop = self.load_images(region_alias)
        xmax, xmin, ymax, ymin = region_crop

        # Define crop size
        max_x_iter = int((xmax - xmin - crop_width) / stride)
        max_y_iter = int((ymax - ymin - crop_height) / stride)
        curr_x_start, curr_y_start = xmin - stride, ymin - stride

        ct = 0  # Initial image

        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = region_crop[0] - stride
            for _ in range(max_x_iter):
                curr_x_start += stride
                x, y = curr_x_start, curr_y_start

                # Crop image and label into sliding window
                img_crop, lbl_crop = self.crop_images(img, lbl, x, y, crop_height, crop_width)
                lbl_out = self.process_labels(lbl_crop, min_sheet_area)

                # Object instances are encoded as unique colors
                # Ensure there is at least 1 ice block in the image
                if len(np.unique(lbl_out)) > 1:
                    img_crop = cv2.resize(img_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    lbl_out = cv2.resize(lbl_out, (1024, 1024), interpolation=cv2.INTER_NEAREST)

                    # Save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # Convert back to images and save
                    self.save_as_images(img_crop, lbl_out, img_dir, lbl_dir, region_alias, x, y, ct)

                    ct += 1  # Move to next image

    def load_images(self, region_alias):
            """
            Load images and labels for the specified region.
            """
            img = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:, :, 0]
            lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:, :, :3]

            # Get extent of region
            region_lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias]))
            region_activation = np.where(region_lbl)

            xmin, xmax = np.min(region_activation[1]), np.max(region_activation[1])
            ymin, ymax = np.min(region_activation[0]), np.max(region_activation[0])
            region_crop = [xmin, xmax, ymin, ymax]

            return img, lbl, region_crop

    @staticmethod
    def process_labels(lbl_crop, min_sheet_area):
        """
        Process the labels by pruning small areas and broken labels.
        """
        lbl_crop_processed = np.zeros_like(lbl_crop).astype(int)
        rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
        struc_data = lbl_crop.view(rgb_type)
        unique_colors = np.unique(struc_data)
        mask_id = 1

        for void_color in unique_colors:
            mask_color = void_color.tolist()
            if mask_color == (0, 0, 0):
                continue

            single_mask = np.all(lbl_crop == mask_color, axis=2)
            if DataGenerator.crop_too_small(single_mask, min_sheet_area) or \
                    DataGenerator.crop_has_broken_label(single_mask):
                continue
            else:
                lbl_crop_processed = np.where(single_mask > 0, mask_id, lbl_crop_processed)
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
        """
        cv2.imwrite(f'{img_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_crop)
        img_png = np.zeros((img_crop.shape[0], img_crop.shape[1], 3))
        img_png[:, :, 0] = img_crop
        img_png[:, :, 1] = img_crop
        img_png[:, :, 2] = img_crop
        for loc in list(zip(*np.where(lbl_out))):
            img_png[loc[0], loc[1], 0] += 100
        cv2.imwrite(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_png)

    @staticmethod
    def crop_has_broken_label(lbl_crop):
        """
        Check if the cropped label has a broken label.
        """
        lbl_probe = lbl_crop.copy()
        lbl_probe[1:(lbl_probe.shape[0] - 1), 1:(lbl_probe.shape[1] - 1)] = 0
        if len(np.unique(lbl_probe)) > 1:
            return True
        else:
            return False

    @staticmethod
    def crop_too_small(crop, min_area):
        """
        Check if the cropped region is too small.
        """
        crop_area = np.sum(np.where(crop > 0, 1, 0))
        if crop_area < min_area:
            return True
        else:
            return False

    def get_directories(self, set_type):
        """
        Get the directories for the specified set type.
        """
        if set_type == "train":
            return self.train_img_path, self.train_lbl_path
        elif set_type == "test":
            return self.test_img_path, self.test_lbl_path
