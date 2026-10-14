import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from Evaluate_Error import evaluat_error
from Evaluation_battery import evaluation_battery


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


def Model_KNN(x_train, y_train, x_test, y_test):


    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    if y_train.ndim == 2:
        y_train_lbl = np.argmax(y_train, axis=1)
        y_test_lbl  = np.argmax(y_test, axis=1)
        num_classes = y_train.shape[1]
    else:
        y_train_lbl = y_train
        y_test_lbl  = y_test
        num_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train_lbl)

    probs = model.predict(X_test)
    predict = (probs >= 0.5).astype(int)
    Eval = evaluat_error(y_test, predict)
    return Eval, predict


def Model_KNN_energy(x_train, y_train, x_test, y_test):


    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    if y_train.ndim == 2:
        y_train_lbl = np.argmax(y_train, axis=1)
        y_test_lbl  = np.argmax(y_test, axis=1)
        num_classes = y_train.shape[1]
    else:
        y_train_lbl = y_train
        y_test_lbl  = y_test
        num_classes = len(np.unique(y_train))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train_lbl)

    probs = model.predict(X_test)
    predict = (probs >= 0.5).astype(int)
    Eval = evaluation_battery(y_test, predict)
    return Eval, predict