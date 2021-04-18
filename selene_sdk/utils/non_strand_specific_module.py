import tensorflow as tf
import torch.nn as nn

from . import _is_lua_trained_model


def _flip(x: tf.Tensor, dim):
    xsize = x.shape
    dim = x.ndim + dim if dim < 0 else dim
    # x = x.contiguous()
    x = tf.reshape(x, [-1, *xsize[dim:]])
    x = tf.reshape(x, [x.shape[0], x.shape[1], -1])[:, getattr(
        tf.experimental.numpy.arange(x.shape[1] - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return tf.reshape(x, [xsize])


class NonStrandSpecific(nn.Module):
    def __init__(self, model, mode="mean"):
        super(NonStrandSpecific, self).__init__()
        self.model = model
        if mode != "mean" and mode != "max":
            raise ValueError("Mode should be one of 'mean' or 'max' but was {0}.".format(mode))
        self.mode = mode
        self.from_lua = _is_lua_trained_model(model)

    def forward(self, my_input):
        if self.from_lua:
            reverse_input = tf.expand_dims(_flip(_flip(tf.squeeze(my_input, 2), 1), 2), 2)
        else:
            reverse_input = _flip(_flip(my_input, 1), 2)
        output = self.model.forward(my_input)
        output_from_rev = self.model.forward(reverse_input)
        if self.mode == "mean":
            return (output + output_from_rev) / 2
        else:
            return tf.math.reduce_max(output, output_from_rev)
