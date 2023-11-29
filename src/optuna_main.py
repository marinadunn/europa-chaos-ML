import optuna
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from utils.optuna_utility.optuna_wrapper import OptunaWrapper
from utils.optuna_utility.objectives import avg_f1_objective, avg_recall_objective, avg_precision_objective
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


# Define main function
def optuna_search(args, obj_fn):
    """
    Perform hyperparameter search using optuna wrapper and command-line args,
    save the best paramaters, save search result plots.

    Args:
        args (argparse.Namespace): Command-line arguments.
        obj_fn (callable): Objective function for optimization.
    """
    # Create wrapper
    optuna_obj = OptunaWrapper(optim_metric=args.metric, trial_count=30, obj_fn=obj_fn)

    # Perform optuna search with the given metric and number of trials
    optuna_obj.optimize()
    optuna_obj.save_best_params()  # Save best params
    optuna_obj.plot_search()  # Plot search results


if __name__ == "__main__":

    # Use command-line arguments for optuna hyperparameter optimization
    parser = argparse.ArgumentParser(description="Run optuna hyperparam search with \
                                                command-line argument for metric.")

    # Add command-line arguments

    # Which metric to optimize
    parser.add_argument("--metric", "-m", type=str, required=True, choices=["f1", "precision", "recall"],
                        help="Which metric to use for optuna hyperparameter optimization. \
                            Options currently include f1, precision, or recall.")

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.metric == "f1":
        obj_fn = avg_f1_objective
    elif args.metric == "recall":
        obj_fn = avg_recall_objective
    elif args.metric == "precision":
        obj_fn = avg_precision_objective
    else:
        print("Metric not supported yet. Consider a different approach.")
        exit(1)

    # Call the main function with the parsed arguments
    optuna_search(args, obj_fn)
