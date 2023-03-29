import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from FeatureSelection.boruta_util import *

X, y = load_iris(return_X_y=True)
X = pd.DataFrame(X)


rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=10, random_state=0)

print(select_features_boruta(X, y, rf, log=False))
# print(select_features_borutashap(X, y, True))


