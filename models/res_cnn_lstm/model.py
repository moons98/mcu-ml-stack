"""Res-CNN-LSTM (DscConv) — Gao et al. 2025, architecture from Supporting Information Table S8.

Layers / output shapes / params per Table S8:
  input (16,6)
  separable_conv1d   (16,32)  242   ReLU
  max_pooling1d      (16,32)  0     pool=2, strides=1, same  (sequence length stays 16)
  separable_conv1d_1 (16,32)  1152  (no activation; Add -> Activation follows)
  add                (16,32)  0     sepconv_1 + max_pooling1d
  activation         (16,32)  0     ReLU
  max_pooling1d_1    (16,32)  0     pool=2, strides=1, same
  lstm               (16)     3136
  dense              (4)      68    softmax
  total 4598
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_res_cnn_lstm(input_shape=(16, 6), n_classes=4, filters=32, lstm_units=16):
    inp = keras.Input(shape=input_shape, name="input_1")
    x = layers.SeparableConv1D(filters, 3, padding="same", activation="relu", name="separable_conv1d")(inp)
    p1 = layers.MaxPooling1D(pool_size=2, strides=1, padding="same", name="max_pooling1d")(x)
    x = layers.SeparableConv1D(filters, 3, padding="same", name="separable_conv1d_1")(p1)
    x = layers.Add(name="add")([x, p1])
    x = layers.Activation("relu", name="activation")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding="same", name="max_pooling1d_1")(x)
    x = layers.LSTM(lstm_units, name="lstm")(x)
    out = layers.Dense(n_classes, activation="softmax", name="dense")(x)
    return keras.Model(inp, out, name="res_cnn_lstm_dsc")


if __name__ == "__main__":
    m = build_res_cnn_lstm()
    m.summary()
    assert m.count_params() == 4598, m.count_params()
    print("OK: params == 4598 (Table S8)")
