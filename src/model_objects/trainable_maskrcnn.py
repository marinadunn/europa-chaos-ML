import torch
from src.data_generator import DataGenerator
from src.model_objects.abstract_maskrcnn import AbstractMaskRCNN
from src.dataset import EuropaIceBlockDataset, EuropaIceBlockMetaset, EuropaIceBlockDatasetTorch
from src.info.file_structure import IMG_TEST_PATH, IMG_TRAIN_PATH, LBL_TEST_PATH, LBL_TRAIN_PATH
from src.info.model_info import TRANSFER_TRAINED_MODEL_FOLDER
from src.utility.custom import get_transform, get_transform_torch
import src.utility.utils as utils
from src.utility.engine import train_one_epoch

class TrainableMaskRCNN(AbstractMaskRCNN):
    def __init__(self, arch_id, num_classes):
        self.arch_id = arch_id
        self.num_classes = num_classes
        # Main purpose is to train a model than save the weights
        super().__init__()
        self.data_gen = DataGenerator()

    def train_model_given_regions(self, num_epochs, train_regions, test_regions, stride=64, crop_size=256):
        self.load_and_set_model(
            self.arch_id, # Arch_ID
            self.num_classes,
            trainable_layers=3,
            type="v2",
            dropout_factor=0.2)

        data_gen_params = {
            "train_regions": train_regions,
            "test_regions": test_regions,
            "crop_height": crop_size,
            "crop_width": crop_size,
            "stride": stride,
            "min_sheet_area": 200
        }

        self.data_gen.pruned_sliding_crops_experiment(
            data_gen_params["train_regions"],
            data_gen_params["test_regions"],
            data_gen_params["crop_height"],
            data_gen_params["crop_width"],
            data_gen_params["stride"],
            data_gen_params["min_sheet_area"]
        )

        dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=True))  

        bs = 1
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)

        params = [p for p in self.model.parameters() if p.requires_grad]
        learning_rate = 0.001
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        #How is LR schedularl working? Gamma value??  what the change is??
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000) #remove lr step, is 3000 step size insane?

        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=(100))

    # Mimics the structure of objective with the exact values suggested by optuna
    def train_model(self, num_epochs):
        self.load_and_set_model(
            self.arch_id, # Arch_ID
            self.num_classes,
            trainable_layers=3,
            type="v2",
            dropout_factor=0.2)

        crop_size = 384 #512
        stride = 16 # 32

        data_gen_params = {
            "train_regions": ["A", "B", "C", "Co", "D", "F", "G", "H", "I"],
            "test_regions": ["hh", "ii"],
            "crop_height": crop_size,
            "crop_width": crop_size,
            "stride": stride,
            "min_sheet_area": 200
        }
        # data_gen_params = {
        #     "train_regions": ["A", "aa", "B", "bb", "C", "Co", "D", "dd", "E", "ee", "F", "ff", "G", "gg", "H", "I", "jj", "kk"],
        #     "test_regions": ["hh", "ii"],
        #     "crop_height": crop_size,
        #     "crop_width": crop_size,
        #     "stride": stride,
        #     "min_sheet_area": 100 #50s
        # }
        self.data_gen.pruned_sliding_crops_experiment(
            data_gen_params["train_regions"],
            data_gen_params["test_regions"],
            data_gen_params["crop_height"],
            data_gen_params["crop_width"],
            data_gen_params["stride"],
            data_gen_params["min_sheet_area"]
        )

        # dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=True))  
        dataset = EuropaIceBlockDataset('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform(train=True))  
        # dataset = EuropaIceBlockDatasetTorch('./', IMG_TRAIN_PATH, LBL_TRAIN_PATH,  get_transform_torch(train=True))  

        bs = 2
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)

        params = [p for p in self.model.parameters() if p.requires_grad]
        learning_rate = 0.001
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        #How is LR schedularl working? Gamma value??  what the change is??
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000) #remove lr step, is 3000 step size insane?

        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=(100))

    def save_model(self):
        model_name = f"type_{self.arch_id}_transfer_trained_model.pth"
        model_path = f"{TRANSFER_TRAINED_MODEL_FOLDER}/{model_name}"
        print("Saving model weights...")
        torch.save(self.model.state_dict(), model_path)