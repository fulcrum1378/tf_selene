from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six

from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


@tf_export("Module")
class Module(tracking.AutoTrackable):
    # AutoTrackable adds object attributes that users will not expect us to
    # include when flattening (these reference dependencies reachable via other
    # object attributes).
    _TF_MODULE_IGNORED_PROPERTIES = frozenset((
        "_self_unconditional_checkpoint_dependencies",
        "_self_unconditional_dependency_names"
    ))

    def __init__(self, name=None):
        if name is None:
            name = camel_to_snake(type(self).__name__)
        else:
            if not valid_identifier(name):
                raise ValueError(
                    "%r is not a valid module name. Module names must be valid Python "
                    "identifiers (e.g. a valid class name)." % name)

        self._name = name
        if tf2.enabled():
            with ops.name_scope_v2(name) as scope_name:
                self._name_scope = ops.name_scope_v2(scope_name)
        else:
            with ops.name_scope(name, skip_on_eager=False) as scope_name:
                self._scope_name = scope_name

    @property
    def name(self):
        return self._name

    @property
    def name_scope(self):
        if tf2.enabled():
            return self._name_scope
        else:
            # In TF1 name_scope is not re-entrant in eager so we cannot memoize it.
            return ops.name_scope(self._scope_name, skip_on_eager=False)

    @property
    def variables(self):
        return tuple(self._flatten(predicate=_is_variable, expand_composites=True))

    @property
    def trainable_variables(self):
        return tuple(
            self._flatten(predicate=_is_trainable_variable, expand_composites=True))

    @property
    def submodules(self):
        return tuple(self._flatten(predicate=_is_module))

    def _flatten(self,
                 recursive=True,
                 predicate=None,
                 attribute_traversal_key=None,
                 with_path=False,
                 expand_composites=False):
        if predicate is None:
            predicate = lambda _: True

        return _flatten_module(
            self,
            recursive=recursive,
            predicate=predicate,
            attributes_to_ignore=self._TF_MODULE_IGNORED_PROPERTIES,
            attribute_traversal_key=attribute_traversal_key,
            with_path=with_path,
            expand_composites=expand_composites)

    @classmethod
    def with_name_scope(cls, method):
        def method_with_name_scope(self, *args, **kwargs):
            with self.name_scope:
                return method(self, *args, **kwargs)
        return tf_decorator.make_decorator(method, method_with_name_scope)


def _is_variable(obj):
    return isinstance(obj, variables.Variable)


def _is_trainable_variable(obj):
    return _is_variable(obj) and getattr(obj, "trainable", False)


def _is_module(obj):
    return isinstance(obj, Module)


_CAMEL_TO_SNAKE_R = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")


def valid_identifier(name):
    return bool(_VALID_IDENTIFIER.match(name))


def camel_to_snake(value):
    return _CAMEL_TO_SNAKE_R.sub(r"_\1", value).lower()


def _flatten_module(module,
                    recursive,
                    predicate,
                    attribute_traversal_key,
                    attributes_to_ignore,
                    with_path,
                    expand_composites,
                    module_path=(),
                    seen=None):
    if seen is None:
        seen = set([id(module)])

    module_dict = vars(module)
    submodules = []

    for key in sorted(module_dict, key=attribute_traversal_key):
        if key in attributes_to_ignore:
            continue

        prop = module_dict[key]
        try:
            leaves = nest.flatten_with_tuple_paths(
                prop, expand_composites=expand_composites)
        except Exception as cause:  # pylint: disable=broad-except
            six.raise_from(
                ValueError(
                    "Error processing property {!r} of {!r}".format(key, prop)),
                cause)

        for leaf_path, leaf in leaves:
            leaf_path = (key,) + leaf_path

            if not with_path:
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)

            if predicate(leaf):
                if with_path:
                    yield module_path + leaf_path, leaf
                else:
                    yield leaf

            if recursive and _is_module(leaf):
                # Walk direct properties first then recurse.
                submodules.append((module_path + leaf_path, leaf))

    for submodule_path, submodule in submodules:
        subvalues = _flatten_module(
            submodule,
            recursive=recursive,
            predicate=predicate,
            attribute_traversal_key=attribute_traversal_key,
            attributes_to_ignore=submodule._TF_MODULE_IGNORED_PROPERTIES,  # pylint: disable=protected-access
            with_path=with_path,
            expand_composites=expand_composites,
            module_path=submodule_path,
            seen=seen)

        for subvalue in subvalues:
            # Predicate is already tested for these values.
            yield subvalue
