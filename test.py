import tensorflow as tf
import torch as ch

arr = [[1, 2, 3], [3, 2, 1]]
print(ch.Tensor(arr).size(1))
print(tf.constant(arr).shape)
