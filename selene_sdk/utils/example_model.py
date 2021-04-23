from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class DeeperDeepSEA(tf.Module):
    def __init__(self, sequence_length: int, n_targets: int):
        super(DeeperDeepSEA, self).__init__()
        conv_kernel_size, pool_kernel_size = 8, 4

        self.conv_net = tf.keras.Sequential()
        self.conv_net.add(tf.keras.layers.Conv1D(320, conv_kernel_size))  # input: 4
        self.conv_net.add(tf.keras.layers.ReLU())  # all "inplace"s are True
        self.conv_net.add(tf.keras.layers.Conv1D(320, conv_kernel_size))  # input: 320
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.MaxPool1D(strides=pool_kernel_size))  # ALL kernel_size=pool_kernel_size
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # num_features: 320

        self.conv_net.add(tf.keras.layers.Conv1D(480, conv_kernel_size))  # input: 320
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.Conv1D(480, conv_kernel_size))  # input: 480
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.MaxPool1D(strides=pool_kernel_size))
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # num_features: 480
        self.conv_net.add(tf.keras.layers.Dropout(0.2))

        self.conv_net.add(tf.keras.layers.Conv1D(960, conv_kernel_size))  # input: 480
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.Conv1D(960, conv_kernel_size))  # input: 960
        self.conv_net.add(tf.keras.layers.ReLU())
        self.conv_net.add(tf.keras.layers.BatchNormalization())  # num_features: 960
        self.conv_net.add(tf.keras.layers.Dropout(0.2))

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size) - reduce_by) / pool_kernel_size)
            - reduce_by)

        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.ELU())  # 960 * self._n_channels, n_targets
        self.classifier.add(tf.keras.layers.ReLU())
        self.classifier.add(tf.keras.layers.BatchNormalization(axis=n_targets))
        self.classifier.add(tf.keras.layers.ELU())  # n_targets, n_targets
        # self.classifier.add(tf.keras.activations.sigmoid)

    def forward(self, x: tf.Tensor) -> tf.keras.Sequential:
        out = self.conv_net(x)
        return self.classifier(tf.reshape(out, [out.shape[0], 960 * self._n_channels]))


def criterion() -> tf.keras.losses.Loss:
    return tf.keras.losses.BinaryCrossentropy()


def get_optimizer(learning_rate) -> Tuple:
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])
    return tfa.optimizers.SGDW, {
        "learning_rate": learning_rate,
        "weight_decay": lambda: 1e-6 * schedule(step),  # 1e-6,
        "momentum": 0.9
    }
