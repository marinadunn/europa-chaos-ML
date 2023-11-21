from torch.utils.data import DataLoader

from src.data_generator import DataGenerator
from src.dataset import EuropaIceBlockDataset
from src.config import (IMG_TEST_PATH, IMG_TRAIN_PATH,
                        LBL_TEST_PATH, LBL_TRAIN_PATH,
                        device)
from src.utils.custom import get_transform
import src.utils.utils as utils
from src.utils.optuna_utility.models import define_mrcnn_model_01, define_mrcnn_model_02
from src.utils.engine import train_one_epoch
from src.model_objects.rcnn_model import MaskRCNN
from src.utils.optuna_utility.evaluation import MaskRCNNOutputEvaluator

# Create data generator
data_gen = DataGenerator()

def get_data_loaders(train_transform,
                     test_transform,
                     batch_size,
                     img_train_path,
                     lbl_train_path,
                     img_test_path,
                     lbl_test_path
                     ):
    """
    Create DataLoader instances for training and testing.

    Args:
        train_transform: Transformation for training data.
        test_transform: Transformation for testing data.
        batch_size (int): Batch size.
        img_train_path (str): Path to training images.
        lbl_train_path (str): Path to training labels.
        img_test_path (str): Path to testing images.
        lbl_test_path (str): Path to testing labels.

    Returns:
        tuple: DataLoader instances for training and testing.
    """
    dataset = EuropaIceBlockDataset('../', img_train_path, lbl_train_path, train_transform)
    dataset_test = EuropaIceBlockDataset('../', img_test_path, lbl_test_path, test_transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)

    return dataset, dataset_test, data_loader, data_loader_test


def train_model(trial, model, optimizer, data_loader, device, num_epochs):
    """
    Train the given model.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
        model: Model to be trained.
        optimizer: Model optimizer.
        data_loader: DataLoader for training data.
        device: Device for training.
        num_epochs (int): Number of training epochs.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=(num_epochs + 10))


def objective(trial):
    """
    Objective function for optuna.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: IoU score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [200, 250, 300, 350])
    stride = trial.suggest_categorical("stride", [25, 50, 75, 100])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8]) # 8 default but was getting memory errors

    data_gen_params = {
        "train_regions": ["Co"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": 50
    }

    data_gen.pruned_sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Get data loaders
    dataset, dataset_test, data_loader, data_loader_test = get_data_loaders(get_transform(train=False),
                                                                            get_transform(train=False),
                                                                            batch_size,
                                                                            IMG_TRAIN_PATH,
                                                                            LBL_TRAIN_PATH,
                                                                            IMG_TEST_PATH,
                                                                            LBL_TEST_PATH
                                                                            )

    num_classes = 2
    model = define_mrcnn_model_01(trial, num_classes)
    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = torch.optim.SGD(params, lr=learning_rate)

    num_epochs = 100
    # Train model
    train_model(trial, model, optimizer, data_loader, device, num_epochs)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    # Set model to wrapper object
    mask_model.set_model(model)

    true_segms, pred_segms = mask_model.calc_cumulative_threshold_matrix(data_loader_test)
    iou_score = evaluator.calc_cumulative_threshold_iou(true_segms, pred_segms)

    return iou_score  # Return the metric to be optimized


def avg_f1_objective(trial):
    """
    Objective function for optuna to maximize average F1 score.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

    Returns:
        float: Average F1 score.
    """
    # Define hyperparameters to be optimized
    crop_size = trial.suggest_categorical("crop_size", [200])
    stride = trial.suggest_categorical("stride", [25])
    batch_size = trial.suggest_categorical("batch_size", [2])
    num_epochs = trial.suggest_int('num_epochs', 7, 13)

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                          "D", "dd", "E", "ee", "F", "ff",
                          "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": 50
    }

    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Get data loaders
    dataset, dataset_test, data_loader, data_loader_test = get_data_loaders(get_transform(train=False),
                                                                            get_transform(train=False),
                                                                            batch_size,
                                                                            IMG_TRAIN_PATH,
                                                                            LBL_TRAIN_PATH,
                                                                            IMG_TEST_PATH,
                                                                            LBL_TEST_PATH
                                                                            )

    num_classes = 2
    model = define_mrcnn_model_02(trial, num_classes)
    # Set model to device
    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 0.006174236796209435
    optimizer = torch.optim.SGD(params, lr=learning_rate)

    # Train model
    train_model(trial, model, optimizer, data_loader, device, num_epochs)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    # Set model to wrapper object
    mask_model.set_model(model)

    dataset_avg_f1_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, 20)

    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_f1,_ = evaluator.calc_min_iou_avg_f1(true_label, thresh_sweep, min_iou=0.3)
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
    crop_size = trial.suggest_categorical("crop_size", [200])
    stride = trial.suggest_categorical("stride", [32])
    batch_size = trial.suggest_categorical("batch_size", [2])
    num_epochs = trial.suggest_int('num_epochs', 7, 14)

    data_gen_params = {
        "train_regions": ["A", "aa", "B", "bb", "C", "Co",
                          "D", "dd", "E", "ee", "F", "ff",
                          "G", "gg", "H", "I", "jj", "kk"],
        "test_regions": ["hh", "ii"],
        "crop_size": crop_size,
        "stride": stride,
        "min_sheet_area": 200
    }

    data_gen.sliding_crops_experiment(
        data_gen_params["train_regions"],
        data_gen_params["test_regions"],
        data_gen_params["crop_size"],
        data_gen_params["stride"],
        data_gen_params["min_sheet_area"]
    )

    # Get data loaders
    dataset, dataset_test, data_loader, data_loader_test = get_data_loaders(get_transform(train=False),
                                                                            get_transform(train=False),
                                                                            batch_size,
                                                                            IMG_TRAIN_PATH,
                                                                            LBL_TRAIN_PATH,
                                                                            IMG_TEST_PATH,
                                                                            LBL_TEST_PATH
                                                                            )

    num_classes = 2
    model = define_mrcnn_model_02(trial, num_classes)
    # Set model to device
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train model
    train_model(trial, model, optimizer, data_loader, device, num_epochs)

    # Evaluate model - transfers model to wrapper object
    mask_model = MaskRCNN()
    evaluator = MaskRCNNOutputEvaluator()
    # Set model to wrapper object
    mask_model.set_model(model)

    dataset_avg_recall_scores = []
    thresh_sweeps = mask_model.get_dataset_thresh_sweeps(dataset_test, 20)

    for thresh_sweep_pair in thresh_sweeps:
        thresh_sweep = thresh_sweep_pair[0]
        true_label = thresh_sweep_pair[1]
        avg_recall,_ = evaluator.calc_min_iou_avg_recall(true_label, thresh_sweep, min_iou=0.3)
        dataset_avg_recall_scores.append(avg_recall)

    dataset_avg_recall = 0

    if len(dataset_avg_recall_scores) != 0:
        dataset_avg_recall = sum(dataset_avg_recall_scores) / len(dataset_avg_recall_scores)

    return dataset_avg_recall  # Return the metric to be optimized