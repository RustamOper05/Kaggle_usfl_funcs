import optuna


def visual_history_of_study(study):
    #optim history
    optuna.visualization.plot_optimization_history(study)
    #slice plot
    optuna.visualization.plot_slice(study)
    #hyperparam importances
    optuna.visualization.plot_param_importances(study)
