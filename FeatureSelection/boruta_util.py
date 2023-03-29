import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from BorutaShap import BorutaShap
from numba import njit
from typing import Dict, List


# @njit
def select_features_boruta(X: pd.DataFrame, y, model, verbose=5, random_state=0, log=True) -> Dict:
    feat_selector = BorutaPy(model, n_estimators='auto', verbose=verbose, random_state=random_state)
    feat_selector.fit(X.values, y)

    important = list(X.columns[feat_selector.support_])
    tentative = list(X.columns[feat_selector.support_weak_])
    unimportant = list(X.columns[~(feat_selector.support_ | feat_selector.support_weak_)])

    if log:
        # Check selected features
        print(feat_selector.support_)
        # Select the chosen features from our dataframe.
        selected = X[:, feat_selector.support_]
        print("")
        print("Selected Feature Matrix Shape")
        print(selected.shape)

    columns = {
        'important': important,
        'tentative': tentative,
        'unimportant': unimportant
    }

    return columns

# @njit
# def select_features_borutashap(X, y, classification, model=None, verbose=True, log=True):
#
#     feat_selector = BorutaShap(importance_measure='shap',
#                               classification=classification)
#     if model:
#         feat_selector = BorutaShap(model=model, importance_measure='shap',
#                                    classification=classification)
#
#     feat_selector.fit(X=X, y=y, n_trials=100, sample=False,
#                          train_or_test='test', normalize=True,
#                          verbose=verbose)
#
#     if log:
#         feat_selector.plot(which_features='all')
#         feat_selector.Subset()
