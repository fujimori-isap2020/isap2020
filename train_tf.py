import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re
import h5py
import pathlib
from sklearn.model_selection import train_test_split

def create_model():
    inputs = layers.Input((7,))
    x = layers.Dense(5, activation="relu")(inputs)
    x = layers.Dense(5, activation="relu")(x)
    x = layers.Dense(4, activation="linear")(x)
    return tf.keras.models.Model(inputs, x)

def train(dataset, dataset_answer):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, dataset_answer, test_size=0.2, random_state=42)
    print(Y_train.shape)
    print(Y_test.shape)
    model = create_model()
    #loss = tf.keras.losses.mean_squared_error()
    #acc = tf.keras.metrics.binary_accuracy()
    optim = tf.keras.optimizers.Adam()

    # train
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=['binary_accuracy'])
    result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=128)

    # eval
    val_loss, val_acc = model.evaluate(X_test, Y_test, batch_size=128)
    print(val_loss, val_acc)

    return result


def main():
    #dataset = ''
    with h5py.File('degree.hdf5', 'r') as f:
        #dataset = f.get('rxpowers').value
        dataset = f['rxpowers'].value
    with h5py.File('positions.hdf5', mode='r') as f:
        dataset_answer = f['position'].value

    
    dataset = dataset.reshape(7, dataset.shape[1]*dataset.shape[2]).T
    dataset_answer = dataset_answer.reshape(4, dataset_answer.shape[1]*dataset_answer.shape[2]).T
    
    train_result = train(dataset, dataset_answer)



if __name__ == "__main__":
    main()