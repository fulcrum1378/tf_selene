"""
This module specifies the MultiModelWrapper class, currently intended for
use through Selene's API (as opposed to the CLI).

Loads multiple models and concatenates their outputs.
"""
import torch
import torch.nn as nn


class MultiModelWrapper(nn.Module):

    def __init__(self, sub_models, concat_dim=1):
        super(MultiModelWrapper, self).__init__()
        self.sub_models = sub_models
        self._concat_dim = concat_dim

    def cuda(self):
        for sm in self.sub_models:
            sm.cuda()

    def eval(self):
        for sm in self.sub_models:
            sm.eval()

    def forward(self, x):
        return torch.cat(
            [sm(x) for sm in self.sub_models], self._concat_dim)
