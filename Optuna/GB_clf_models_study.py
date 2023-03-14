import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def xgb_objective_clf(trial):
    from xgboost import XGBClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        'max_depth': trial.suggest_int('depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.1),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        'eta': trial.suggest_float('eta', 1e-8, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }

    model = XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    pred_labels = model.predict(X_test)
    loss = log_loss(y_test, pred_labels)
    return loss


def lgbm_objective_clf(trial):
    from lightgbm import LGBMClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    param = {
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'device': 'gpu',
        'max_depth': trial.suggest_int('depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.1)
    }

    model = LGBMClassifier(**param)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    pred_labels = model.predict(X_test)
    loss = log_loss(y_test, pred_labels)
    return loss


def cb_objective_clf(trial):
    from catboost import CatBoostClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    param = {
        'iterations': 3000,
        'learning_rate': trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True),
        'depth': trial.suggest_int("depth", 2, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        'bootstrap_type': trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        'random_strength': trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 10.0),
        'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        'od_wait': trial.suggest_int("od_wait", 10, 50),
        'task_type': "GPU",
        'devices': "0:1"
    }

    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    pred_labels = model.predict(X_test)
    loss = log_loss(y_test, pred_labels)
    return loss


def Objective(trial):
    import torch
    from sklearn.model_selection import KFold
    from pytorch_tabnet.tab_model import TabNetClassifier
    # pip install --quiet pytorch-tabnet

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 1, 64, step=4)
    n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
    gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 3)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params = dict(
        n_d=n_da,
        n_a=n_da,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type=mask_type, n_shared=n_shared,
        scheduler_params=dict(
            mode="min",
            patience=trial.suggest_int("patienceScheduler", low=3, high=10),
            min_lr=1e-8,
            factor=0.5,
        ),
         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
         verbose=0,
         device_name=DEVICE
     )

    kf = KFold(n_splits=4, random_state=42, shuffle=True)
    CV_score_array = []

    for train_index, test_index in kf.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        regressor = TabNetClassifier(**tabnet_params)
        regressor.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            patience=50,
            max_epochs=3000,
            eval_metric=['log_loss']
        )
        CV_score_array.append(regressor.best_cost)

    avg = np.mean(CV_score_array)
    return avg
