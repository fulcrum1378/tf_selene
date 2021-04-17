import numpy as np
from torch.optim import SGD
from torch.nn import BatchNorm1d, BCELoss, Conv1d, Dropout, Linear, MaxPool1d, Module, ReLU, Sequential, Sigmoid


class DeeperDeepSEA(Module):
    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = Sequential(
            Conv1d(4, 320, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            Conv1d(320, 320, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            BatchNorm1d(320),

            Conv1d(320, 480, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            Conv1d(480, 480, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            BatchNorm1d(480),
            Dropout(p=0.2),

            Conv1d(480, 960, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            Conv1d(960, 960, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            BatchNorm1d(960),
            Dropout(p=0.2))

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = Sequential(
            Linear(960 * self._n_channels, n_targets),
            ReLU(inplace=True),
            BatchNorm1d(n_targets),
            Linear(n_targets, n_targets),
            Sigmoid())

    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


def criterion():
    return BCELoss()


def get_optimizer(lr):
    return SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9}
