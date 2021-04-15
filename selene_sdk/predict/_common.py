import math

import numpy as np
import torch
from torch.autograd import Variable

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


def predict(model, batch_sequences, use_cuda=False):
    inputs = torch.Tensor(batch_sequences)
    if use_cuda:
        inputs = inputs.cuda()
    with torch.no_grad():
        inputs = Variable(inputs)
        if _is_lua_trained_model(model):
            outputs = model.forward(
                inputs.transpose(1, 2).contiguous().unsqueeze_(2))
        else:
            outputs = model.forward(inputs.transpose(1, 2))
        return outputs.data.cpu().numpy()


def _pad_sequence(sequence, to_length, unknown_base):
    diff = (to_length - len(sequence)) / 2
    pad_l = int(np.floor(diff))
    pad_r = math.ceil(diff)
    sequence = ((unknown_base * pad_l) + sequence + (unknown_base * pad_r))
    return str.upper(sequence)


def _truncate_sequence(sequence, to_length):
    start = int((len(sequence) - to_length) // 2)
    end = int(start + to_length)
    return str.upper(sequence[start:end])
