import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional, Concatenate, Add
from tensorflow.keras.optimizers import Adam
from Evaluate_Error import evaluat_error
from Evaluation_battery import  evaluation_battery


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        score = tf.matmul(x, self.W)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * x, axis=1)

def preprocess_x(X):
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    n, t, f = X.shape
    X2 = X.reshape(n * t, f)
    df = pd.DataFrame(X2)

    for c in df.columns:
        if df[c].dtype == object:
            try:
                dt = pd.to_datetime(df[c], format='%d-%m-%Y %H:%M')
                df[c] = dt.astype('int64') // 10**9
            except:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.fillna(0)
    return df.to_numpy(dtype=np.float32).reshape(n, t, f)

def preprocess_y(y):
    y = pd.DataFrame(y)
    for c in y.columns:
        if y[c].dtype == object:
            y[c] = pd.to_numeric(y[c], errors='coerce')
    y = y.fillna(0)
    return y.to_numpy(dtype=np.float32)

def Model_ARA_DRNN(x_train, y_train, x_test, y_test, sol=None):
    if sol is None:
        sol=[5,0.01,5]
    x_train = preprocess_x(x_train)
    x_test = preprocess_x(x_test)
    y_train = preprocess_y(y_train)
    y_test = preprocess_y(y_test)
    t = x_train.shape[1]
    f = x_train.shape[2]
    inp = Input(shape=(t, f))
    outs = []
    x = inp

    for _ in range(3):
        r = Bidirectional(GRU(64, return_sequences=True))(x)
        outs.append(r)
        x = Concatenate()(outs)

    inp_proj = Dense(x.shape[-1])(inp)
    res = Add()([x, inp_proj])

    att = TemporalAttention()(res)

    x = Dense(int(sol[0]), activation='relu')(att)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    if y_train.shape[1] == 1:
        out = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        out = Dense(y_train.shape[1], activation='softmax')(x)
        loss = 'categorical_crossentropy'
    model = Model(inp, out)
    model.compile(optimizer=Adam(float(sol[1])), loss=loss, metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        epochs=int(sol[2]),
        batch_size=8,
        validation_data=(x_test, y_test),
        verbose=1
    )

    pred = model.predict(x_test)
    Eval = evaluat_error(y_test, pred)
    return Eval, pred



def Model_ARA_DRNN_energy(x_train, y_train, x_test, y_test, sol=None):
    if sol is None:
        sol=[5,0.01,5]
    x_train = preprocess_x(x_train)
    x_test = preprocess_x(x_test)
    y_train = preprocess_y(y_train)
    y_test = preprocess_y(y_test)
    t = x_train.shape[1]
    f = x_train.shape[2]
    inp = Input(shape=(t, f))
    outs = []
    x = inp

    for _ in range(3):
        r = Bidirectional(GRU(64, return_sequences=True))(x)
        outs.append(r)
        x = Concatenate()(outs)

    inp_proj = Dense(x.shape[-1])(inp)
    res = Add()([x, inp_proj])

    att = TemporalAttention()(res)

    x = Dense(int(sol[0]), activation='relu')(att)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    if y_train.shape[1] == 1:
        out = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        out = Dense(y_train.shape[1], activation='softmax')(x)
        loss = 'categorical_crossentropy'
    model = Model(inp, out)
    model.compile(optimizer=Adam(float(sol[1])), loss=loss, metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        epochs=int(sol[2]),
        batch_size=8,
        validation_data=(x_test, y_test),
        verbose=1
    )

    pred = model.predict(x_test)
    Eval = evaluation_battery(y_test, pred)
    return Eval, pred