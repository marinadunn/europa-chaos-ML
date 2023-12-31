import optuna
# uses matplotlib backend rather than plotly
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_parallel_coordinate,
                                             plot_param_importances
                                             )
import numpy as np
import matplotlib.pyplot as plt
import sys

from utils.file_utils import make_dir, append_input_to_file, clear_file
from config import OPTUNA_OUTPUT_PATH


class OptunaWrapper():
    """
    A simple wrapper for Optuna to streamline optimization tasks.

    Attributes:
        MAXIMAL_METRICS (list): List of metrics to maximize during optimization.
        MINIMAL_METRICS (list): List of metrics to minimize during optimization.
        PARAM_SAVE_FILE (str): Filename for saving the best parameters.

    Methods:
        __init__: Initializes the OptunaWrapper.
        optimize: Runs the optimization process.
        get_best_trial: Returns the best trial.
        get_best_params: Returns the best parameters.
        get_best_score: Returns the best score.
        save_best_params: Saves the best parameters to a file.
    """
    MAXIMAL_METRICS = ["precision", "recall", "iou", "f1"]
    MINIMAL_METRICS = ["loss"]
    PARAM_SAVE_FILE = "optuna_best_params.txt"

    def __init__(self, optim_metric, trial_count, obj_fn):
        """
        Initialize the OptunaWrapper.

        Args:
            optim_metric (str): Metric to optimize.
            trial_count (int): Number of trials in the optimization.
            obj_fn (callable): Objective function for optimization.
        """
        # Ensure optuna output directory exists
        make_dir(OPTUNA_OUTPUT_PATH)
        self.optim_metric = optim_metric
        self.trial_count = trial_count
        self.obj_fn = obj_fn
        self.optimized = False
        self.study = None

        # Can also use a pruner for early stopping
        # self.pruner = optuna.pruners.HyperbandPruner(reduction_factor=4)

        # Create hyperparameter tuning "study" session
        # Use "maximize" direction for metrics like accuracy, precision, or recall
        if optim_metric in OptunaWrapper.MAXIMAL_METRICS:
            self.study = optuna.create_study(direction="maximize")
        # Use "minimize" direction for metrics like loss
        elif optim_metric in OptunaWrapper.MINIMAL_METRICS:
            self.study = optuna.create_study(direction="minimize")
        else:
            print("OptunaWrapper does not support this metric yet. Consider a different approach.")

    def optimize(self):
        """Run the optimization process."""
        self.study.optimize(self.obj_fn, self.trial_count)
        self.optimized=True

    def get_best_trial(self):
        """Return the best trial."""
        return self.study.best_trial

    def get_best_params(self):
        """Return the best parameters."""
        best_trial = self.study.best_trial
        return best_trial.params

    def get_best_score(self):
        """Return the best score."""
        best_trial = self.study.best_trial
        return best_trial.value

    def save_best_params(self):
        """Save the best parameters to a file."""
        save_file_path = f"{OPTUNA_OUTPUT_PATH}/{OptunaWrapper.PARAM_SAVE_FILE}"
        clear_file(save_file_path)

        trial_info = f"Total trial Count: {self.trial_count}"
        append_input_to_file(save_file_path, trial_info)

        best_score = self.get_best_score()
        score_info = f"Best trial {self.optim_metric} score: {best_score}"
        append_input_to_file(save_file_path, score_info)

        intro_line = f"-----Best Params-----"
        append_input_to_file(save_file_path, intro_line)

        best_params = self.get_best_params()
        for key, value in best_params.items():
            param_info = f"{key}: {value}"
            append_input_to_file(save_file_path, param_info)

    def plot_search(self):
        """Plot the optimization process."""
        if self.optimized:

            # Plot optimization history of all trials in a study
            plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(f"{OPTUNA_OUTPUT_PATH}/optimization_history.png",
                        bbox_inches='tight', dpi=300)

            # Plot the high-dimensional parameter relationships in a study
            plot_parallel_coordinate(self.study)
            plt.tight_layout()
            plt.savefig(f"{OPTUNA_OUTPUT_PATH}/parallel_coordinate.png",
                        bbox_inches='tight', dpi=300)

            # Plot hyperparameter importances
            plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(f"{OPTUNA_OUTPUT_PATH}/param_importances.png",
                        bbox_inches='tight', dpi=300)
