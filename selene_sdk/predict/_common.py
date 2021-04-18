import math

import numpy as np
import tensorflow as tf

from ..utils import _is_lua_trained_model


def get_reverse_complement(allele, complementary_base_dict):
    if allele == '*' or allele == '-' or len(allele) == 0:
        return '*'
    a_complement = []
    for a in allele:
        a_complement.append(complementary_base_dict[a])
    return ''.join(list(reversed(a_complement)))


def get_reverse_complement_encoding(allele_encoding, bases_arr, complementary_base_dict):
    base_ixs = {b: i for (i, b) in enumerate(bases_arr)}
    complement_indices = [
        base_ixs[complementary_base_dict[b]] for b in bases_arr]
    return allele_encoding[:, complement_indices][::-1, :]


def predict(model: tf.Module, batch_sequences, use_cuda: bool = False):
    inputs = tf.constant(batch_sequences)
    if use_cuda:
        inputs = inputs.cuda()
    inputs = tf.Variable(inputs, trainable=False)
    if _is_lua_trained_model(model):
        outputs = model.forward(tf.expand_dims(tf.transpose(inputs)), 2)  # , perm=[1, 2] (tf)
    else:
        outputs = model.forward(tf.transpose(inputs))  # , perm=[1, 2] (tf)
    return outputs.data.cpu().numpy()


def _pad_sequence(sequence: str, to_length: int, unknown_base: str) -> str:
    diff = (to_length - len(sequence)) / 2
    pad_l = int(np.floor(diff))
    pad_r = math.ceil(diff)
    sequence = ((unknown_base * pad_l) + sequence + (unknown_base * pad_r))
    return str.upper(sequence)


def _truncate_sequence(sequence, to_length) -> str:
    start = int((len(sequence) - to_length) // 2)
    end = int(start + to_length)
    return str.upper(sequence[start:end])
