import optuna
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from data_generator import DataGenerator
from dataset import EuropaIceBlockDataset
from config import (IMG_TEST_PATH,
                    IMG_TRAIN_PATH,
                    LBL_TEST_PATH,
                    LBL_TRAIN_PATH,
                    device)
import config as config
from utils.custom import get_transform
from utils.utils import collate_fn
from utils.optuna_utility.models import define_mrcnn_model_01, define_mrcnn_model_02
from utils.engine import train_one_epoch
from model_objects.rcnn_model import MaskRCNN
from utils.optuna_utility.evaluation import MaskRCNNOutputEvaluator


# Define number of classes
num_classes = 2

# Create data generator
data_gen = DataGenerator()

# Define objective functions for optuna hyperparameter optimization

def objective(trial):
    """
    Objective function for optuna.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: IoU score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [128, 150, 200, 250, 300, 350, 400, 450, 512])
    stride = trial.suggest_categorical("stride", [8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])  # Default is 8, but can produce memory errors
    min_sheet_area = trial.suggest_categorical("min_sheet_area", [4, 8, 25, 50])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 200, step=50)
    opt = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    data_gen_params = {
        "train_regions": ["Co"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": min_sheet_area
    }

    data_gen.pruned_sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Define datasets
    dataset = EuropaIceBlockDataset('', IMG_TRAIN_PATH, LBL_TRAIN_PATH, get_transform(train=False))
    dataset_test = EuropaIceBlockDataset('', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=True))

    # Define PyTorch data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define model
    model = define_mrcnn_model_02(trial, num_classes)

    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]

    # Select the optimizer based on the provided argument
    if opt == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    else:
        print("Invalid optimizer choice. Please choose different optimizer.")
        return

    # Train model
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, epoch, print_freq=(num_epochs + 10), device=config.device)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()

    # Set model to wrapper object
    mask_model.set_model(model)

    # Calculate IoU
    true_segms, pred_segms = mask_model.calc_cumulative_threshold_matrix(data_loader_test)
    iou_score = evaluator.calc_cumulative_threshold_iou(true_segms, pred_segms)

    return iou_score  # Return the metric to be optimized


def avg_f1_objective(trial):
    """
    Objective function for optuna to maximize average F1 score.

    Note: The optimal hyperparameters for this objective function
    were found to be:
        - learning_rate: 0.001,
        - batch_size: 1
        - crop_size: 250
        - stride: 64
        - optimizer: Adam
        - trainable backbone layers: 3
        - num_epochs: 15

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: Average F1 score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [128, 150, 200, 250, 300, 350, 400, 450, 512])
    stride = trial.suggest_categorical("stride", [8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])  # Default is 8, but can produce memory errors
    min_sheet_area = trial.suggest_categorical("min_sheet_area", [4, 8, 25, 50])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 200, step=50)
    opt = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                          "D", "dd", "E", "ee", "F", "ff",
                          "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": min_sheet_area
    }

    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Define datasets
    dataset = EuropaIceBlockDataset('', IMG_TRAIN_PATH, LBL_TRAIN_PATH, get_transform(train=False))
    dataset_test = EuropaIceBlockDataset('', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=True))

    # Define PyTorch data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define model
    model = define_mrcnn_model_02(trial, num_classes)

    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]

    # Select the optimizer based on the provided argument
    if opt == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    else:
        print("Invalid optimizer choice. Please choose different optimizer.")
        return

    # Train model
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, epoch, print_freq=(num_epochs + 10), device=config.device)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()

    # Set model to wrapper object
    mask_model.set_model(model)

    dataset_avg_f1_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, thresh_count=20)

    # Calculate average F1
    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_f1, f1_scores = evaluator.calc_min_iou_avg_f1(true_label, thresh_sweep, min_iou=0.1)
        dataset_avg_f1_scores.append(avg_f1)

    dataset_avg_f1 = 0

    if len(dataset_avg_f1_scores) != 0:
        dataset_avg_f1 = sum(dataset_avg_f1_scores) / len(dataset_avg_f1_scores)

    return dataset_avg_f1  # Return the metric to be optimized


def avg_recall_objective(trial):
    """
    Objective function for optuna to maximize average recall.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: Average recall score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [128, 150, 200, 250, 300, 350, 400, 450, 512])
    stride = trial.suggest_categorical("stride", [8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])  # Default is 8, but can produce memory errors
    min_sheet_area = trial.suggest_categorical("min_sheet_area", [4, 8, 25, 50])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 200, step=50)
    opt = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                          "D", "dd", "E", "ee", "F", "ff",
                          "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": min_sheet_area
    }

    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Define datasets
    dataset = EuropaIceBlockDataset('', IMG_TRAIN_PATH, LBL_TRAIN_PATH, get_transform(train=False))
    dataset_test = EuropaIceBlockDataset('', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=True))

    # Define PyTorch data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define model
    model = define_mrcnn_model_02(trial, num_classes)

    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]

    # Select the optimizer based on the provided argument
    if opt == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    else:
        print("Invalid optimizer choice. Please choose different optimizer.")
        return

    # Train model
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, epoch, print_freq=(num_epochs + 10), device=config.device)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    # Set model to wrapper object
    mask_model.set_model(model)

    dataset_avg_recall_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, thresh_count=20)

    # Calculate average recall
    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_recall, recall_scores = evaluator.calc_min_iou_avg_recall(true_label, thresh_sweep, min_iou=0.1)
        dataset_avg_recall_scores.append(avg_recall)

    dataset_avg_recall = 0

    if len(dataset_avg_recall_scores) != 0:
        dataset_avg_recall = sum(dataset_avg_recall_scores) / len(dataset_avg_recall_scores)

    return dataset_avg_recall  # Return the metric to be optimized


def avg_precision_objective(trial):
    """
    Objective function for optuna to maximize average precision.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: Average precision score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [128, 150, 200, 250, 300, 350, 400, 450, 512])
    stride = trial.suggest_categorical("stride", [8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])  # Default is 8, but can produce memory errors
    min_sheet_area = trial.suggest_categorical("min_sheet_area", [4, 8, 25, 50])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 200, step=50)
    opt = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                          "D", "dd", "E", "ee", "F", "ff",
                          "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": min_sheet_area
    }

    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Define datasets
    dataset = EuropaIceBlockDataset('', IMG_TRAIN_PATH, LBL_TRAIN_PATH, get_transform(train=False))
    dataset_test = EuropaIceBlockDataset('', IMG_TEST_PATH, LBL_TEST_PATH, get_transform(train=True))

    # Define PyTorch data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define model
    model = define_mrcnn_model_02(trial, num_classes)

    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]

    # Select the optimizer based on the provided argument
    if opt == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    else:
        print("Invalid optimizer choice. Please choose different optimizer.")
        return

    # Train model
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, epoch, print_freq=(num_epochs + 10), device=config.device)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    # Set model to wrapper object
    mask_model.set_model(model)

    dataset_avg_precision_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, thresh_count=20)

    # Calculate average precision
    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_precision, precision_scores = evaluator.calc_min_iou_avg_precision(true_label, thresh_sweep, min_iou=0.1)
        dataset_avg_precision_scores.append(avg_precision)

    dataset_avg_precision = 0

    if len(dataset_avg_precision_scores) != 0:
        dataset_avg_precision = sum(dataset_avg_precision_scores) / len(dataset_avg_precision_scores)

    return dataset_avg_precision  # Return the metric to be optimized
