from collections import OrderedDict
from typing import Iterator, Optional, Set, Tuple

import tensorflow as tf


class EnhancedModule(tf.Module):
    def __init__(self, name=None):
        tf.Module.__init__(self, name)
        self._parameters = OrderedDict()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = ''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
