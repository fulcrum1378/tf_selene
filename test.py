import selene_sdk as s

x = int

print(x("12") + 1)

import torch as ch
import tensorflow as tf

# print(ch.arange(55, 77))
# print(tf.experimental.numpy.arange(55, 77))

# my_data = [[1., -1.], [1., -1.]]
# print(ch.max(ch.Tensor(my_data)))  # torch.Tensor
# print(tf.math.reduce_max(tf.constant(my_data)))  # tensorflow.python.framework.ops.EagerTensor

# x = ch.Tensor([1.], requires_grad=True)
# with ch.no_grad():
# y = x * 2
# print(y.requires_grad)

# x = tf.Tensor([[1., -1.], [1., -1.]])
# tf.stop_gradient(x)
# print(x.get_shape())
