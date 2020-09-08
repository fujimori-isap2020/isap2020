#import tensorflow as tf
#import tensorflow.keras.layers as layers
import numpy as np
import os
import re
import h5py
import pathlib

import const


def create_model():
    inputs = layers.Input((784,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(10, activation="softmax")(x)
    return tf.keras.models.Model(inputs, x)


def train():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 784) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 784) / 255.0

    model = create_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    optim = tf.keras.optimizers.Adam()

    # train
    model.compile(optimizer=optim, loss=loss, metrics=[acc])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128)

    # eval
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
    print(val_loss, val_acc)


def read_dataset():
    each_dataset_name = list()
    path = pathlib.Path(const.dataset_dir)
    for p in path.iterdir():
        match = re.match(const.dataset_pattern, str(p.name))
        if match:
            each_dataset_name.append(int(match.group(1)))

    with h5py.File('degree0.hdf5', 'r') as f:
        for k in f:
            print(k)
    print('hello!')
    return each_dataset_name


def main():
    print(read_dataset())


if __name__ == "__main__":
    main()
