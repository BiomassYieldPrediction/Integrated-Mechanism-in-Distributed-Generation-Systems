import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
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


def Model_DNN(x_train, y_train, x_test, y_test):


    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test  = np.asarray(y_test, dtype=np.float32)

    y_train = np.nan_to_num(y_train)
    y_test  = np.nan_to_num(y_test)

    input_shape = (X_train.shape[1],)

    binary_classification = (
        y_train.ndim == 1 or
        (y_train.ndim == 2 and y_train.shape[1] == 1)
    )
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))


    model.add(Dense(y_train.shape[1], activation='softmax'))
    loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )
    probs = model.predict(X_test)
    predict = (probs >= 0.5).astype(int)
    Eval = evaluat_error(y_test, predict)
    return Eval, predict


def Model_DNN_energy(x_train, y_train, x_test, y_test):


    X_train = preprocess_tabular_data(x_train)
    X_test  = preprocess_tabular_data(x_test)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test  = np.asarray(y_test, dtype=np.float32)

    y_train = np.nan_to_num(y_train)
    y_test  = np.nan_to_num(y_test)

    input_shape = (X_train.shape[1],)

    binary_classification = (
        y_train.ndim == 1 or
        (y_train.ndim == 2 and y_train.shape[1] == 1)
    )
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))


    model.add(Dense(y_train.shape[1], activation='softmax'))
    loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )
    probs = model.predict(X_test)
    predict = (probs >= 0.5).astype(int)
    Eval = evaluation_battery(y_test, predict)
    return Eval, predict