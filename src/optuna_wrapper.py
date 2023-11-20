import optuna
from utils.file_utils import make_dir, append_input_to_file, clear_file
from config import OPTUNA_OUTPUT_PATH


class OptunaWrapper():
    # Functions as a simple wrapper for optuna
    MAXIMAL_METRICS = ["precision", "recall", "iou", "f1"]
    MINIMAL_METRICS = ["loss"]
    PARAM_SAVE_FILE = "optuna_best_params.txt"

    def __init__(self, optim_metric, trial_count, obj_fn):
        make_dir(OPTUNA_OUTPUT_PATH)
        self.optim_metric = optim_metric
        self.study = None
        self.obj_fn = obj_fn
        self.trial_count = trial_count
        self.optimized = False
        if optim_metric in OptunaWrapper.MAXIMAL_METRICS:
            self.study = optuna.create_study(direction="maximize")
        elif optim_metric in OptunaWrapper.MINIMAL_METRICS:
            self.study = optuna.create_study(direction="minimize")
        else:
            print("Optunawrapper does not support this metric yet, may not be using the right approach for given metric")

    def optimize(self):
        self.study.optimize(self.obj_fn, self.trial_count)
        self.optimized=True

    def get_best_trial(self):
        return self.study.best_trial

    def get_best_params(self):
        best_trial = self.study.best_trial
        return best_trial.params

    def get_best_score(self):
        best_trial = self.study.best_trial
        return best_trial.value

    def save_best_params(self):
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