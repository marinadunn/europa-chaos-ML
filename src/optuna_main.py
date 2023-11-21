from optuna_wrapper import OptunaWrapper
from src.utils.optuna_utility.objectives import objective, avg_f1_objective, avg_recall_objective

def optuna_search():
    optuna_obj = OptunaWrapper("recall", 10, avg_recall_objective)  # 10 trials, recall as metric
    optuna_obj.optimize()
    optuna_obj.save_best_params()
    optuna_obj.plot_search()


if __name__ == "__main__":
    # perform search using wrapper, save best params, plot search results
    optuna_search()