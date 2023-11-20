import os
import numpy as np
from PIL import Image
import cv2
import config as config
from utils.file_utils import clear_and_remake_directory


class DataGenerator():
    # Generates data based on Europa dataset structure
    def __init__(self):
        # obtain path names from config file
        self.processed_path = config.PROCESSED_DATA_PATH
        self.train_img_path = config.IMG_TRAIN_PATH
        self.train_lbl_path = config.LBL_TRAIN_PATH
        self.test_img_path = config.IMG_TEST_PATH
        self.test_lbl_path = config.LBL_TEST_PATH
        self.prev_settings = None
        self.prev_function = None
        self.config = config

    def reset_processed_data_dir(self):
        # clear and remake processed data directory, subdirectories recursively
        clear_and_remake_directory(self.processed_path)
        os.makedirs(self.train_img_path)
        os.makedirs(self.train_lbl_path)
        os.makedirs(self.test_img_path)
        os.makedirs(self.test_lbl_path)

    def same_method_and_settings(self, curr_function, curr_settings):
        # check if current function and settings are the same as previous
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

        # open images with PIL; convert to numpy arrays
        img = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:, :, 0]
        lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:, :, :3]

        imgs = []
        lbls = []

        # Get extent of region
        region_lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias]))
        region_activation = np.where(region_lbl)

        xmin = np.min(region_activation[1])
        xmax = np.max(region_activation[1])
        ymin = np.min(region_activation[0])
        ymax = np.max(region_activation[0])
        region_crop = [xmin, xmax, ymin, ymax]

        # define crop size
        max_x_iter = int((xmax - xmin) - crop_width) + 1  # This has a boundary problem, can extend past full img size
        max_y_iter = int((ymax - ymin) - crop_height) + 1  # This has a boundary problem, can extend past full img size

        curr_y_start = ymin - crop_height

        ct = 0  # initial image

        for _ in range(max_y_iter):
            curr_y_start += crop_height
            curr_x_start = xmin - crop_width
            for _ in range(max_x_iter):
                curr_x_start += crop_width
                x = curr_x_start
                y = curr_y_start

                # crop image and label into sliding window
                img_crop = np.copy(img[y: y + crop_height, x: x + crop_width])
                lbl_raw_crop = np.copy(lbl[y: y + crop_height, x: x + crop_width])
                lbl_crop = np.zeros_like(img_crop).astype(int)
                rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
                struc_data = lbl_raw_crop.view(rgb_type)
                unique_colors = np.unique(struc_data)
                mask_id = 1

                for void_color in unique_colors:
                    mask_color = void_color.tolist()
                    if mask_color == (0, 0, 0):
                        continue

                    single_mask = np.all(lbl_raw_crop == mask_color, axis=2)
                    if DataGenerator.crop_too_small(single_mask, min_sheet_area):
                        continue
                    if DataGenerator.crop_has_broken_label(single_mask):
                        continue
                    else:
                        lbl_crop = np.where(single_mask > 0, mask_id, lbl_crop)
                        mask_id += 1

                # Object instances are encoded as unique colors
                # Ensures there is at least 1 ice block in image
                if len(np.unique(lbl_crop)) > 1:
                    lbl_out = lbl_crop.copy()
                    # add to lists
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
        Generate sliding crops of specified size and stride, prune partial
        labels.
        """

        img_dir = None
        lbl_dir = None

        # specify train, test directories
        if set_type == "train":
            img_dir = self.train_img_path
            lbl_dir = self.train_lbl_path
        elif set_type == "test":
            img_dir = self.test_img_path
            lbl_dir = self.test_lbl_path

        # open images with PIL; convert to numpy arrays
        img = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:, :, 0]
        lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:, :, :3]

        # Get extent of region
        region_lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias]))
        region_activation = np.where(region_lbl)

        xmin = np.min(region_activation[1])
        xmax = np.max(region_activation[1])
        ymin = np.min(region_activation[0])
        ymax = np.max(region_activation[0])
        region_crop = [xmin, xmax, ymin, ymax]

        # define crop size
        max_x_iter = int(((xmax - xmin) - crop_width) / stride) + 1
        max_y_iter = int(((ymax - ymin) - crop_height) / stride) + 1

        curr_x_start = xmin - stride
        curr_y_start = ymin - stride

        ct = 0  # initial image

        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = xmin - stride
            for _ in range(max_x_iter):
                curr_x_start += stride
                x = curr_x_start
                y = curr_y_start

                # crop image and label into sliding window
                img_crop = np.copy(img[y: y + crop_height, x: x + crop_width])
                lbl_raw_crop = np.copy(lbl[y: y + crop_height, x: x + crop_width])
                lbl_crop = np.zeros_like(img_crop).astype(int)
                rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
                struc_data = lbl_raw_crop.view(rgb_type)
                unique_colors = np.unique(struc_data)
                mask_id = 1

                for void_color in unique_colors:
                    mask_color = void_color.tolist()
                    if mask_color == (0, 0, 0):
                        continue

                    single_mask = np.all(lbl_raw_crop == mask_color, axis=2)
                    if DataGenerator.crop_too_small(single_mask, min_sheet_area):
                        continue
                    if DataGenerator.crop_has_broken_label(single_mask):
                        continue
                    else:
                        lbl_crop = np.where(single_mask > 0, mask_id, lbl_crop)
                        mask_id += 1

                # Object instances are encoded as unique colors
                # Ensures there is at least 1 ice block in image
                if len(np.unique(lbl_crop)) > 1:  # remove background (0)
                    img_crop = cv2.resize(img_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    lbl_crop = cv2.resize(lbl_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    lbl_out = lbl_crop.copy()

                    # save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # convert back to images and save
                    cv2.imwrite(f'{img_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_crop)
                    img_png = np.zeros((img_crop.shape[0], img_crop.shape[1], 3))
                    # copy to all channels
                    img_png[:, :, 0] = img_crop
                    img_png[:, :, 1] = img_crop
                    img_png[:, :, 2] = img_crop
                    for loc in list(zip(*np.where(lbl_out))):
                        img_png[loc[0], loc[1], 0] += 100
                    cv2.imwrite(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_png)

                    ct += 1  # move to next image

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

        img_dir = None
        lbl_dir = None

        if set_type == "train":
            img_dir = self.train_img_path
            lbl_dir = self.train_lbl_path
        elif set_type == "test":
            img_dir = self.test_img_path
            lbl_dir = self.test_lbl_path

        # open images with PIL; convert to numpy arrays
        img = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_FILE_MAP[region_alias]))[:, :, 0]
        lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias]))[:, :, :3]

        # Get extent of region
        region_lbl = np.array(Image.open(self.config.CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias]))
        region_activation = np.where(region_lbl)

        xmin = np.min(region_activation[1])
        xmax = np.max(region_activation[1])
        ymin = np.min(region_activation[0])
        ymax = np.max(region_activation[0])
        region_crop = [xmin, xmax, ymin, ymax]

        # define crop size
        max_x_iter = int(((xmax - xmin) - crop_width) / stride)
        max_y_iter = int(((ymax - ymin) - crop_height) / stride)
        curr_x_start = xmin - stride
        curr_y_start = ymin - stride

        ct = 0  # initial image

        for _ in range(max_y_iter):
            curr_y_start += stride
            curr_x_start = xmin - stride
            for _ in range(max_x_iter):
                curr_x_start += stride
                x = curr_x_start
                y = curr_y_start

                # crop image and label into sliding window
                img_crop = np.copy(img[y: y + crop_height, x: x + crop_width])
                lbl_raw_crop = np.copy(lbl[y: y + crop_height, x: x + crop_width])
                lbl_crop = np.zeros_like(img_crop).astype(int)
                rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
                struc_data = lbl_raw_crop.view(rgb_type)
                unique_colors = np.unique(struc_data)
                mask_id = 1

                for void_color in unique_colors:
                    mask_color = void_color.tolist()
                    if mask_color == (0, 0, 0):
                        continue

                    single_mask = np.all(lbl_raw_crop == mask_color, axis=2)
                    if DataGenerator.crop_too_small(single_mask, min_sheet_area):
                        continue
                    lbl_crop = np.where(single_mask > 0, mask_id, lbl_crop)
                    mask_id += 1

                # Object instances are encoded as unique colors
                # Ensures there is at least 1 ice block in image
                if len(np.unique(lbl_crop)) > 1:  # remove background (0)
                    img_crop = cv2.resize(img_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    lbl_crop = cv2.resize(lbl_crop, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    lbl_out = lbl_crop.copy()

                    # save numpy files
                    np.save(f'{img_dir}/{region_alias}_{x}_{y}_{ct}', img_crop)
                    np.save(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}', lbl_out)

                    # convert back to images and save
                    cv2.imwrite(f'{img_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_crop)
                    img_png = np.zeros((img_crop.shape[0], img_crop.shape[1], 3))
                    # copy to all channels
                    img_png[:, :, 0] = img_crop
                    img_png[:, :, 1] = img_crop
                    img_png[:, :, 2] = img_crop
                    for loc in list(zip(*np.where(lbl_out))):
                        img_png[loc[0], loc[1], 0] += 100
                    cv2.imwrite(f'{lbl_dir}/{region_alias}_{x}_{y}_{ct}_visual.png', img_png)

                    ct += 1  # move to next image

    @staticmethod
    def crop_has_broken_label(lbl_crop):
        lbl_probe = lbl_crop.copy()
        lbl_probe[1:(lbl_probe.shape[0] - 1), 1:(lbl_probe.shape[1] - 1)] = 0
        if len(np.unique(lbl_probe)) > 1:
            return True
        else:
            return False

    @staticmethod
    def crop_too_small(crop, min_area):
        crop_area = np.sum(np.where(crop > 0, 1, 0))
        if crop_area < min_area:
            return True
        else:
            return False
