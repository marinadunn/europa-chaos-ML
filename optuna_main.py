from src.optuna_wrapper import OptunaWrapper
from src.optuna_utility.objectives import objective, avg_f1_objective, avg_recall_objective

def optuna_search():
    optuna_obj = OptunaWrapper("recall", 10, avg_recall_objective)
    optuna_obj.optimize()
    optuna_obj.save_best_params()


if __name__ == "__main__":
    optuna_search()