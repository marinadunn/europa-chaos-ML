import torch

from data_generator import DataGenerator
from model_objects.abstract_maskrcnn import AbstractMaskRCNN
from dataset import EuropaIceBlockDataset
from config import (IMG_TEST_PATH,
                    IMG_TRAIN_PATH,
                    LBL_TEST_PATH, LBL_TRAIN_PATH,
                    TRANSFER_TRAINED_MODEL_FOLDER,
                    device)
from utils.custom import get_transform
import utils.utils as utils
from utils.engine import train_one_epoch

class TrainableMaskRCNN(AbstractMaskRCNN):
    def __init__(self, arch_id, num_classes):
        """
        Initialize the TrainableMaskRCNN.

        Args:
            arch_id (int): Identifier for the model architecture version.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.arch_id = arch_id
        self.num_classes = num_classes
        self.data_gen = DataGenerator()

    def train_model_given_regions(self,
                                  num_epochs,
                                  train_regions,
                                  test_regions,
                                  stride=64,
                                  crop_size=250
                                  ):
        """
        Train the model with specific regions using sliding window approach.

        Args:
            num_epochs (int): Number of training epochs.
            train_regions (list): List of regions for training.
            test_regions (list): List of regions for testing.
            stride (int): Stride for sliding window.
            crop_size (int): Size of the cropped images.
        """
        self.load_and_set_model(self.arch_id,
                                self.num_classes,
                                trainable_layers=3,
                                type="v2",
                                dropout_factor=0.2
                                )

        data_gen_params = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_size": crop_size,
            "stride": stride,
            "min_sheet_area": 50
        }

        self.data_gen.pruned_sliding_crops_experiment(
            data_gen_params["train_regions"],
            data_gen_params["test_regions"],
            data_gen_params["crop_size"],
            data_gen_params["stride"],
            data_gen_params["min_sheet_area"]
        )

        dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,
                                        get_transform(train=True))

        # default batch size is 1
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  shuffle=False,
                                                  collate_fn=utils.collate_fn
                                                  )

        params = [p for p in self.model.parameters() if p.requires_grad]
        learning_rate = 0.001
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, epoch, print_freq=100, device=device)

    # Mimics the structure of objective with the exact values suggested by optuna
    def train_model(self, num_epochs):
        """
        Train the model with default regions and settings.

        Args:
            num_epochs (int): Number of training epochs.
        """
        self.load_and_set_model(self.arch_id,
                                self.num_classes,
                                trainable_layers=3,
                                type="v2",
                                dropout_factor=0.2
                                )

        crop_size = 250
        stride = 64

        data_gen_params = {
            "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                              "D", "dd", "E", "ee", "F", "ff",
                              "G", "gg", "H", "I", "jj", "kk"],
            "test_regions": ["hh", "ii"],
            "crop_size": crop_size,
            "stride": stride,
            "min_sheet_area": 50
        }

        self.data_gen.pruned_sliding_crops_experiment(
            data_gen_params["train_regions"],
            data_gen_params["test_regions"],
            data_gen_params["crop_size"],
            data_gen_params["stride"],
            data_gen_params["min_sheet_area"]
        )

        dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH, get_transform(train=True))

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=utils.collate_fn
                                                  )

        params = [p for p in self.model.parameters() if p.requires_grad]
        learning_rate = 0.001
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, epoch, print_freq=100, device=device)

    def save_model(self):
        """
        Save the trained model.
        """
        model_name = f"type_{self.arch_id}_transfer_trained_model.pth"
        model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/{model_name}"
        print("Saving model weights...")
        torch.save(self.model.state_dict(), model_path)