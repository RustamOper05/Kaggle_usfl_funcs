import numpy as np


def xgboost_clf(X, y, test, k_fold, params, patience):
    from xgboost import XGBClassifier

    modelsXGB = []
    predsXGB = []

    for train_index, test_index in k_fold.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(**params)

        model.fit(X=X_train, y=y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=patience,
                  verbose=100
                  )
        modelsXGB.append(model)
        predsXGB.append(model.predict_proba(test))

    return np.average(np.array(predsXGB), axis=0)[:, 1], predsXGB, modelsXGB


def cb_clf(X, y, test, k_fold, params, patience):
    from catboost import CatBoostClassifier

    modelsCB = []
    predsCB = []

    for train_index, test_index in k_fold.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        model = CatBoostClassifier(**params)

        model.fit(X=X_train, y=y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=patience,
                  verbose=100
                  )
        modelsCB.append(model)
        predsCB.append(model.predict_proba(test))

    return np.average(np.array(predsCB), axis=0)[:, 1], predsCB, modelsCB


def lgb_clf(X, y, test, k_fold, params, patience):
    import lightgbm as lgbm

    modelsLB = []
    predsLB = []

    for train_index, test_index in k_fold.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        model = lgbm.LGBMClassifier(**params)

        model.fit(X=X_train, y=y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=patience,
                  verbose=100
                  )
        modelsLB.append(model)
        predsLB.append(model.predict_proba(test))

    return np.average(np.array(predsLB), axis=0)[:, 1], predsLB, modelsLB


def SLB_clf(X, y, test, k_fold, params):
    from sklearn.ensemble import GradientBoostingClassifier

    modelsSGB = []
    predsSGB = []

    for train_index, test_index in k_fold.split(X, y):
        model = GradientBoostingClassifier(**params)

        model.fit(X, y)
        modelsSGB.append(model)
        predsSGB.append(model.predict_proba(test))

    return np.average(np.array(predsSGB), axis=0)[:, 1], predsSGB, modelsSGB


def tab_net_clf(X, y, test, k_fold, params, patience):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    predictions_array = []
    CV_score_array = []

    for train_index, test_index in k_fold.split(X):
        X_train, X_valid = X[train_index], y[test_index]
        y_train, y_valid = X[train_index], y[test_index]

        clf = TabNetClassifier(**params)

        clf.fit(X_train=X_train, y_train=y_train,
                eval_set=[(X_valid, y_valid)],
                patience=patience,
                max_epochs=3000,
                eval_metric=['log_loss'])

        CV_score_array.append(clf.best_cost)
        predictions_array.append(clf.predict_proba(test.values))

    return np.average(np.array(predictions_array), axis=0).reshape(-1), predictions_array


#####################################################################################################################
# import torch
# import torch.nn as nn
# import torch.utils.data as data_utils
# import torch.nn.functional as F

# device = torch.device('cuda:0')

# class Reg(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(len(X.columns), 32)
#         self.dense1_bn = nn.BatchNorm1d(32)
#         self.fc2 = nn.Linear(32, 16)
#         self.dense2_bn = nn.BatchNorm1d(16)
# #         self.fc3 = nn.Linear(32, 16)
# #         self.dense3_bn = nn.BatchNorm1d(16)
#         self.fc4 = nn.Linear(16, 2)

#     def forward(self, x):
#         x = F.relu(self.dense1_bn(self.fc1(x)))
#         x = F.relu(self.dense2_bn(self.fc2(x)))
# #         x = F.relu(self.dense3_bn(self.fc3(x)))
#         x = F.relu(self.fc4(x))
#         return x

# x_train_tensor = torch.from_numpy(X.values).float().to(device)
# y_train_tensor = torch.from_numpy(y.values).float().to(device)

# model = Reg()
# loss_fn = nn.MSELoss()
# optim =  torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
# model.to(device)

# dataset = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
# train_loader = data_utils.DataLoader(dataset, batch_size=8, shuffle=True)

# losses = []
# for i in range(50):
#     for x_batch, y_batch in train_loader:
#         x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
#         y_batch = y_batch.type(torch.FloatTensor).to(device)
#         y_pred = model(x_batch.float()).to(device)

#         loss = loss_fn(y_pred, y_batch)
#         losses.append(loss.item())

#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#     if(i % 10 == 0):
#         print(f"epochs: {i}......loss:{(losses[-1])}")
# plt.plot(losses)

# predNN = model(torch.tensor(torch.from_numpy(test.values).float().to(device),dtype=torch.float32)).cpu().detach().numpy()
#####################################################################################################################
