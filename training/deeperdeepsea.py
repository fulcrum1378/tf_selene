import numpy as np
import tensorflow as tf


class DeeperDeepSEA(tf.Module):
    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = tf.keras.Sequential()
        self.conv_net.add(tf.keras.layers.Conv1D(320, kernel_size=conv_kernel_size))  # 4
        self.conv_net.add(tf.keras.layers.ReLU())  # ALL: inplace=True
        self.conv_net.add(tf.keras.layers.Conv1D(320, kernel_size=conv_kernel_size))  # 320
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.MaxPool1D(strides=pool_kernel_size))  # kernel_size=pool_kernel_size
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # 320

        self.conv_net.add(tf.keras.layers.Conv1D(480, kernel_size=conv_kernel_size))  # 320
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.Conv1D(480, kernel_size=conv_kernel_size))  # 480
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.MaxPool1D(strides=pool_kernel_size))  # kernel_size=pool_kernel_size
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # 480
        self.conv_net.add(tf.keras.layers.Dropout(0.2))

        self.conv_net.add(tf.keras.layers.Conv1D(960, kernel_size=conv_kernel_size))  # 480
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.Conv1D(960, kernel_size=conv_kernel_size))  # 960
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # 960
        self.conv_net.add(tf.keras.layers.Dropout(0.2))

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.ELU())  # 960 * self._n_channels, n_targets
        self.classifier.add(tf.keras.layers.ReLU())  # inplace=True
        self.classifier.add(tf.keras.layers.BatchNormalization())  # n_targets
        self.classifier.add(tf.keras.layers.ELU())  # n_targets, n_targets
        # self.classifier.add(tf.keras.activations.sigmoid())

    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


def criterion():
    return tf.keras.losses.BinaryCrossentropy()


def get_optimizer(lr):
    return tf.keras.optimizers.SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9}
