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


#Тренировочный цикл для кастомной модели и предикт кастомной модели на pytorch
def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0):
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке.
    :param model: torch.nn.Module - обучаемая модель
    :param train_dataset: torch.utils.data.Dataset - данные для обучения
    :param val_dataset: torch.utils.data.Dataset - данные для оценки качества
    :param criterion: функция потерь для настройки модели
    :param lr: скорость обучения
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param early_stopping_patience: наибольшее количество эпох, в течение которых допускается
        отсутствие улучшения модели, чтобы обучение продолжалось.
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param max_batches_per_epoch_train: максимальное количество итераций на одну эпоху обучения
    :param max_batches_per_epoch_val: максимальное количество итераций на одну эпоху валидации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи
        (по умолчанию torch.utils.data.DataLoader)
    :return: кортеж из двух элементов:
        - среднее значение функции потерь на валидации на лучшей эпохе
        - лучшая модель
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Эпоха {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)



            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    """
    :param model: torch.nn.Module - обученная модель
    :param dataset: torch.utils.data.Dataset - данные для применения модели
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :return: numpy.array размерности len(dataset) x *
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)
