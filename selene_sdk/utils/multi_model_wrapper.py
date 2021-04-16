import tensorflow as tf


class MultiModelWrapper(tf.Module):
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
        return tf.concat([sm(x) for sm in self.sub_models], self._concat_dim)
