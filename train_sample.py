import tensorflow as tf
from tensorflow.keras.layers import *

from scipy.stats import pearsonr
import numpy as np

# 自作モジュール
from config import *
import ga_dataset_tools

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CNNGraduation(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self):
        super(CNNGraduation, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(196, 98, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flat1 = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.hidden3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flat1(x)
        x = self.hidden1(x)
        x = self.bn4(x)
        x = self.hidden2(x)
        return self.hidden3(x)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        # データの準備
        valid_dataset = ga_dataset_tools.GADataset(VALID_DIR)
        ds = valid_dataset.get_tf_dataset2()
        # ds = ds.shuffle(buffer_size=valid_dataset.generation_size * valid_dataset.population_size)
        self.train_images = ds.map(lambda images, labels: images).batch(32)
        # log10適応度
        self.train_labels = np.asarray(list(ds.map(lambda images, labels: log10(labels)).as_numpy_iterator()))

    def on_epoch_end(self, batch, logs={}):
        pred_result = np.asarray(self.model.predict(self.train_images)).flatten()
        print('\nsample[0]:', self.train_labels[30], pred_result[30])
        r, p = pearsonr(self.train_labels, pred_result)
        print(f'corr: {r}')


if __name__ == '__main__':
    # model = CNNGraduation()
    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), input_shape=(196, 98, 1), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        #Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # model.summary()

    result = (np.load('result.npy')).reshape(7,810000).T
    positions = (np.load('positions.npy')).reshape(4,810000).T

    #正規化(時間かかる)
    for i in range(810000):
        before = result[i, :]
        after = (before - before.min())/(before.max()-before.min())
        result[i, :] = after

    #datasetの準備 shuffle
    dataset = np.concatenate([result, positions], 1)
    np.random.shuffle(dataset)

    #trainとtestを8対2で分割
    train, test = np.split(dataset, [int(810000 * 0.8)])


'''
    # データの準備
    train_dataset = ga_dataset_tools.GADataset(TRAIN_DIR)
    ds = train_dataset.get_tf_dataset2()
    ds = ds.map(lambda images, labels: (images, log10(labels)))
    ds = ds.shuffle(buffer_size=train_dataset.generation_size * train_dataset.population_size)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    print(ds)
'''

    # モデルの学習
    corr_cb = LossHistory()
    model.fit(train[:,0:6], train[:,7:10],epochs=NUM_EPOCHS)
