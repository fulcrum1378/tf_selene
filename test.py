import tensorflow as tf
import torch as ch

x = ch.nn.Conv1d(4, 320, kernel_size=8)
print(x.bias)
# print(tf.keras.layers.Conv1D(320, 8))
