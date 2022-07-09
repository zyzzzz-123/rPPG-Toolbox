""" Short functions for data-preprocessing and data-loading. """

import numpy as np
import torch
from collections import OrderedDict

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)


def load_model(model, config):
    if config.NUM_OF_GPU_TRAIN > 1:
        checkpoint = torch.load(config.INFERENCE.MODEL_PATH)
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
    model = model.to(config.DEVICE)
    return model