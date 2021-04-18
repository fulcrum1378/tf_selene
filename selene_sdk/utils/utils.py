from collections import OrderedDict
import logging
import sys
from typing import Dict

import numpy as np
import tensorflow as tf

from .multi_model_wrapper import MultiModelWrapper


def _is_lua_trained_model(model: tf.Module):
    if hasattr(model, 'from_lua'): return model.from_lua
    check_model = model
    if hasattr(model, 'model'):
        check_model = model.model
    elif type(model) == MultiModelWrapper and hasattr(model, 'sub_models'):
        check_model = model.sub_models[0]
    setattr(model, "from_lua", False)
    setattr(check_model, "from_lua", False)
    for m in check_model.submodules:
        if "Conv2d" in m.__class__.__name__:
            setattr(model, "from_lua", True)
            setattr(check_model, "from_lua", True)
    return model.from_lua


def get_indices_and_probabilities(interval_lengths, indices):
    select_interval_lens = np.array(interval_lengths)[indices]
    weights = select_interval_lens / float(np.sum(select_interval_lens))

    keep_indices = []
    for index, weight in enumerate(weights):
        if weight > 1e-10:
            keep_indices.append(indices[index])
    if len(keep_indices) == len(indices):
        return indices, weights.tolist()
    else:
        return get_indices_and_probabilities(interval_lengths, keep_indices)


def load_model_from_state_dict(state_dict: Dict, model: tf.Module):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()

    if len(model_keys) != len(state_dict_keys):
        try:
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            raise ValueError("Loaded state dict does not match the model "
                             "architecture specified - please check that you are "
                             "using the correct architecture file and parameters.\n\n"
                             "{0}".format(e))

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = state_dict[k2]
        try:
            new_state_dict[k1] = value
        except Exception as e:
            raise ValueError(
                "Failed to load weight from module {0} in model weights "
                "into model architecture module {1}. (If module name has "
                "an additional prefix `model.` it is because the model is "
                "wrapped in `selene_sdk.utils.NonStrandSpecific`. This "
                "error was raised because the underlying module does "
                "not match that expected by the loaded model:\n"
                "{2}".format(k2, k1, e))
    model.load_state_dict(new_state_dict)
    return model


def load_features_list(input_path):
    features = []
    with open(input_path, 'r') as file_handle:
        for line in file_handle:
            features.append(line.strip())
    return features


def initialize_logger(output_path, verbosity=2):
    logger = logging.getLogger("selene")
    if len(logger.handlers): return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(output_path)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)

    stdout_formatter = logging.Formatter("%(asctime)s - %(message)s")

    stdout_handle = logging.StreamHandler(sys.stdout)
    stdout_handle.setFormatter(stdout_formatter)
    stdout_handle.setLevel(logging.INFO)
    logger.addHandler(stdout_handle)
