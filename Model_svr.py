import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from Evaluate_Error import evaluat_error
from Evaluation_battery import  evaluation_battery


def preprocess_tabular_data(X):
    X = pd.DataFrame(X)

    for col in X.columns:
        if X[col].dtype == object:
            try:
                dt = pd.to_datetime(X[col], format='%d-%m-%Y %H:%M')
                X[col] = dt.astype('int64') // 10**9
            except:
                X.drop(columns=[col], inplace=True)

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    return X.to_numpy(dtype=np.float32)


def Model_SVR(x_train, y_train, x_test, y_test):

    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    if y_train.ndim == 2 and y_train.shape[1] > 1:
        y_train_lbl = y_train
        y_test_lbl  = y_test
        num_classes = y_train.shape[1]
        multi_class = True
    else:
        y_train_lbl = y_train.ravel()
        y_test_lbl  = y_test.ravel()
        num_classes = 1
        multi_class = False

    base_svr = SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        epsilon=0.1
    )

    if multi_class:
        model = OneVsRestClassifier(base_svr)
    else:
        model = base_svr

    model.fit(X_train, y_train_lbl)
    preds = model.predict(X_test)
    predict = np.where(preds >= 0.5, 1, 0).reshape(-1, 1)
    Eval = evaluat_error(y_test, predict)

    return Eval, predict


def Model_SVR_energy(x_train, y_train, x_test, y_test):

    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    if y_train.ndim == 2 and y_train.shape[1] > 1:
        y_train_lbl = y_train
        y_test_lbl  = y_test
        num_classes = y_train.shape[1]
        multi_class = True
    else:
        y_train_lbl = y_train.ravel()
        y_test_lbl  = y_test.ravel()
        num_classes = 1
        multi_class = False

    base_svr = SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        epsilon=0.1
    )

    if multi_class:
        model = OneVsRestClassifier(base_svr)
    else:
        model = base_svr

    model.fit(X_train, y_train_lbl)
    pred = model.predict(X_test)
    Eval = evaluation_battery(y_test, pred)
    return Eval, pred
