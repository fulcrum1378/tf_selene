import tensorflow as tf
import torch as ch

arr = [[1., 2., 3.], [3., 2., 1.], [4., 5., 6.]]
x, y = ch.Tensor(arr), tf.constant(arr)
print(x[ch.arange(x.size(1) - 1., -1., -1.).long()])
print(tf.gather(y, tf.cast(tf.experimental.numpy.arange(y.shape[1] - 1, -1, -1), tf.int64)))
