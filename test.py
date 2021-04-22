# import tensorflow as tf
# import torch as ch
from typing import Type


class Kir:
    def __init__(self):
        self.verb = "FUCK"


t: Type[Kir] = Kir
print(t().verb)
