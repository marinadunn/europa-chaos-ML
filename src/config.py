import torch
import torchvision
import numpy as np

# set computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 2 classification classes: background and Plate
num_classes = 2
class_names = ['__background__', 'Ice Block']

# Resnet50 Backbone Training variables
weights = "DEFAULT"  # same as COCO_V1
trainable_backbone_layers = 5

# minimum size of the image to be rescaled before feeding it to the backbone
min_size = 200

PROCESSED_DATA_PATH = "../data/processed_data"
IMG_TRAIN_PATH = "../data/processed_data/img_train"
IMG_TEST_PATH = "../data/processed_data/img_test"
LBL_TRAIN_PATH = "../data/processed_data/lbl_train"
LBL_TEST_PATH = "../data/processed_data/lbl_test"

CV_OUTPUT_PATH = "../output/cv_output"
INFO_OUTPUT_PATH = "../output/data_stats"

OPTUNA_OUTPUT_PATH = "../optuna_output"

MODEL_WEIGHTS_PATH = "../models/model_weights"
MODEL_OUTPUT_PATH = "../models"
TRANSFER_TRAINED_MODEL_FOLDER = "../models/model_weights"
MAIN_MODEL_PATH = "../models/model_weights/type_1_transfer_trained_model.pth" # Need to think of abstraction
CUMULATIVE_TEST_MODEL_PATH = "../models/model_weights/cumulative_test_model.pth"

PLOTS_OUTPUT_PATH = "../plots"

# data paths for each chaos region
CHAOS_REGION_ALIAS_TO_FILE_MAP = {
    "aa":"../data/Chaos_aa/image/c11ESREGMAP01_c17ESNERTRM01_c17ESREGMAP01_GalileoSSI_E_reproj.png",
    "bb":"../data/Chaos_bb/image/c11ESREGMAP01_c17ESNERTRM01_c17ESREGMAP01_GalileoSSI_E_reproj.png",
    "Co":"../data/Chaos_Co/image/E6ESDRKLIN01_GalileoSSI_E_reproj.png",
    "dd":"../data/Chaos_dd/image/c11ESREGMAP01_GalileoSSI_E_reproj.png",
    "ee":"../data/Chaos_ee/image/c17ESNERTRM01_GalileoSSI_E_reproj.png",
    "ff":"../data/Chaos_ff/image/c17ESREGMAP01_GalileoSSI_E_reproj.png",
    "gg":"../data/Chaos_gg/image/c17ESREGMAP01_GalileoSSI_E_reproj.png",
    "hh":"../data/Chaos_hh/image/c17ESNERTRM01_GalileoSSI_E_reproj.png",
    "ii":"../data/Chaos_ii/image/c17ESNERTRM01_GalileoSSI_E_reproj.png",
    "jj": "../data/Chaos_jj/image/c17ESNERTRM01_GalileoSSI_E_reproj.png",
    "kk":"../data/Chaos_kk/image/c17ESNERTRM01_GalileoSSI_E_reproj.png",
    "A":"../data/Chaos_A/image/c15ESREGMAP02_GalileoSSI_E_reproj.png",
    "B":"../data/Chaos_B/image/c15ESREGMAP02_GalileoSSI_E_reproj.png",
    "C":"../data/Chaos_C/image/c15ESREGMAP02_GalileoSSI_E_reproj.png",
    "D":"../data/Chaos_D/image/c15ESREGMAP02_GalileoSSI_E_reproj.png",
    "E":"../data/Chaos_E/image/c15ESREGMAP02_GalileoSSI_E_reproj.png",
    "F":"../data/Chaos_F/image/c17ESREGMAP02_GalileoSSI_E_reproj.png",
    "G":"../data/Chaos_G/image/c17ESREGMAP02_GalileoSSI_E_reproj.png",
    "H":"../data/Chaos_H/image/c17ESREGMAP02_GalileoSSI_E_reproj.png",
    "I":"../data/Chaos_I/image/c17ESREGMAP02_GalileoSSI_E_reproj.png"
}

CHAOS_REGION_ALIAS_TO_LABEL_MAP = {
    "aa":"../data/Chaos_aa/label/Chaos_aa_mask.png",
    "bb":"../data/Chaos_bb/label/Chaos_bb_mask.png",
    "Co":"../data/Chaos_Co/label/Chaos_Co_mask.png",
    "dd":"../data/Chaos_dd/label/Chaos_dd_mask.png",
    "ee":"../data/Chaos_ee/label/Chaos_ee_mask.png",
    "ff":"../data/Chaos_ff/label/Chaos_ff_mask.png",
    "gg":"../data/Chaos_gg/label/Chaos_gg_mask.png",
    "hh":"../data/Chaos_hh/label/Chaos_hh_mask.png",
    "ii":"../data/Chaos_ii/label/Chaos_ii_mask.png",
    "jj":"../data/Chaos_jj/label/Chaos_jj_mask.png",
    "kk":"../data/Chaos_kk/label/Chaos_kk_mask.png",
    "A":"../data/Chaos_A/label/Chaos_A_mask.png",
    "B":"../data/Chaos_B/label/Chaos_B_mask.png",
    "C":"../data/Chaos_C/label/Chaos_C_mask.png",
    "D":"../data/Chaos_D/label/Chaos_D_mask.png",
    "E":"../data/Chaos_E/label/Chaos_E_mask.png",
    "F":"../data/Chaos_F/label/Chaos_F_mask.png",
    "G":"../data/Chaos_G/label/Chaos_G_mask.png",
    "H":"../data/Chaos_H/label/Chaos_H_mask.png",
    "I":"../data/Chaos_I/label/Chaos_I_mask.png"
}

CHAOS_REGION_ALIAS_TO_PLATE_LABEL_MAP = {
    "A": "data/chaos_data/Chaos_A/label/Chaos_A_plates_mask.png",
    "aa": "data/chaos_data/Chaos_aa/label/Chaos_aa_plates_mask.png",
    "B": "data/chaos_data/Chaos_B/label/Chaos_B_plates_mask.png",
    "bb": "data/chaos_data/Chaos_bb/label/Chaos_bb_plates_mask.png",
    "C": "data/chaos_data/Chaos_C/label/Chaos_C_plates_mask.png",
    "Co": "data/chaos_data/Chaos_Co/label/Chaos_Co_plates_mask.png",
    "D": "data/chaos_data/Chaos_D/label/Chaos_D_plates_mask.png",
    "dd": "data/chaos_data/Chaos_dd/label/Chaos_dd_plates_mask.png",
    "E": "data/chaos_data/Chaos_E/label/Chaos_E_plates_mask.png",
    "ee": "data/chaos_data/Chaos_ee/label/Chaos_ee_plates_mask.png",
    "F": "data/chaos_data/Chaos_F/label/Chaos_F_plates_mask.png",
    "ff": "data/chaos_data/Chaos_ff/label/Chaos_ff_plates_mask.png",
    "G": "data/chaos_data/Chaos_G/label/Chaos_G_plates_mask.png",
    "gg": "data/chaos_data/Chaos_gg/label/Chaos_gg_plates_mask.png",
    "H": "data/chaos_data/Chaos_H/label/Chaos_H_plates_mask.png",
    "hh": "data/chaos_data/Chaos_hh/label/Chaos_hh_plates_mask.png",
    "I": "data/chaos_data/Chaos_I/label/Chaos_I_plates_mask.png",
    "ii": "data/chaos_data/Chaos_ii/label/Chaos_ii_plates_mask.png",
    "jj": "data/chaos_data/Chaos_jj/label/Chaos_jj_plates_mask.png",
    "kk": "data/chaos_data/Chaos_kk/label/Chaos_kk_plates_mask.png"
}

CHAOS_REGION_ALIAS_TO_REGION_MAP = {
    "aa":"../data/Chaos_aa/label/Chaos_aa_region_mask.png",
    "bb":"../data/Chaos_bb/label/Chaos_bb_region_mask.png",
    "Co":"../data/Chaos_Co/label/Chaos_Co_region_mask.png",
    "dd":"../data/Chaos_dd/label/Chaos_dd_region_mask.png",
    "ee":"../data/Chaos_ee/label/Chaos_ee_region_mask.png",
    "ff":"../data/Chaos_ff/label/Chaos_ff_region_mask.png",
    "gg":"../data/Chaos_gg/label/Chaos_gg_region_mask.png",
    "hh":"../data/Chaos_hh/label/Chaos_hh_region_mask.png",
    "ii":"../data/Chaos_ii/label/Chaos_ii_region_mask.png",
    "jj":"../data/Chaos_jj/label/Chaos_jj_region_mask.png",
    "kk":"../data/Chaos_kk/label/Chaos_kk_region_mask.png",
    "A":"../data/Chaos_A/label/Chaos_A_region_mask.png",
    "B":"../data/Chaos_B/label/Chaos_B_region_mask.png",
    "C":"../data/Chaos_C/label/Chaos_C_region_mask.png",
    "D":"../data/Chaos_D/label/Chaos_D_region_mask.png",
    "E":"../data/Chaos_E/label/Chaos_E_region_mask.png",
    "F":"../data/Chaos_F/label/Chaos_F_region_mask.png",
    "G":"../data/Chaos_G/label/Chaos_G_region_mask.png",
    "H":"../data/Chaos_H/label/Chaos_H_region_mask.png",
    "I":"../data/Chaos_I/label/Chaos_I_region_mask.png"
}


# Image resolutions (meters/pixel)
CHAOS_REGION_RESOLUTION_MAP = {
    "A": 229.0,
    "aa": 222.0,  # Largely in c17ESREGMAP01
    "B": 229.0,
    "bb": 214.0,  # half in c17ESNERTRM01 210.0 and 11ESREGMAP01 218.0
    "C": 229.0,
    "Co": 179.0,
    "D": 229.0,
    "dd": 218.0,
    "E": 229.0,
    "ee": 210.0,
    "F": 215.0,
    "ff": 222.0,
    "G": 215.0,
    "gg": 222.0,
    "H": 215.0,
    "hh": 210.0,
    "I": 215.0,
    "ii": 210.0,
    "jj": 210.0,
    "kk": 210.0
}

# Fixing global random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)