from collections import namedtuple
import os
import re
import six
from typing import Dict
import warnings
import yaml

SCIENTIFIC_NOTATION_REGEXP = r"^[\-\+]?(\d+\.?\d*|\d*\.?\d+)?[eE][\-\+]?\d+$"
IS_INITIALIZED = False

_BaseProxy = namedtuple("_BaseProxy", ["callable", "positionals", "keywords", "yaml_src"])


class _Proxy(_BaseProxy):
    __slots__ = []

    def __hash__(self):
        return hash(id(self))

    def bind(self, **kwargs):
        for k in kwargs:
            if k not in self.keywords:
                self.keywords[k] = kwargs[k]

    def pop(self, key):
        return self.keywords.pop(key)


def _do_not_recurse(value):
    return value


def _instantiate_proxy_tuple(proxy, bindings=None):
    if proxy in bindings:
        return bindings[proxy]
    else:
        if proxy.callable == _do_not_recurse:
            obj = proxy.keywords['value']
        else:
            if len(proxy.positionals) > 0:
                raise NotImplementedError('positional arguments not yet supported in proxy instantiation')
            kwargs = dict((k, instantiate(v, bindings))
                          for k, v in six.iteritems(proxy.keywords))
            obj = proxy.callable(**kwargs)
        try:
            obj.yaml_src = proxy.yaml_src
        except AttributeError:
            pass
        bindings[proxy] = obj
        return bindings[proxy]


def _preprocess(string, environ=None):
    if environ is None: environ = {}
    split = string.split('${')
    rval = [split[0]]
    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before '
                             'end of string or next ${ in "' + string + '"')

        varname = subsplit[0]
        val = (environ[varname] if varname in environ
               else os.environ[varname])
        rval.append(val)
        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)
    return rval


def instantiate(proxy, bindings=None):
    if bindings is None: bindings = {}
    if isinstance(proxy, _Proxy):
        return _instantiate_proxy_tuple(proxy, bindings)
    elif isinstance(proxy, dict):
        return dict((instantiate(k, bindings), instantiate(v, bindings))
                    for k, v in six.iteritems(proxy))
    elif isinstance(proxy, list):
        return [instantiate(v, bindings) for v in proxy]
    elif isinstance(proxy, six.string_types):
        return _preprocess(proxy)
    else:
        return proxy


def load(stream) -> Dict:
    global IS_INITIALIZED
    if not IS_INITIALIZED: _initialize()
    if isinstance(stream, six.string_types):
        string = stream
    else:
        string = stream.read()
    return yaml.load(string, Loader=yaml.SafeLoader)


def load_path(path: str) -> Dict:
    with open(path, 'r') as f:
        content = ''.join(f.readlines())
    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str, got " + str(type(content)))
    return load(content)


def _try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    module_name = '.'.join(components[:-1])
    try:
        exec("import {0}".format(module_name))
    except ImportError as e:
        pieces = module_name.split('.')
        str_e = str(e)
        found = True in [piece.find(str(e)) != -1 for piece in pieces]

        if found:
            raise ImportError("Could not import {0}; ImportError was {1}".format(module_name, str_e))
        else:
            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                module_name = '.'.join(pcomponents[:j])
                try:
                    exec("import {0}".format(module_name))
                except Exception:
                    base_msg = "Could not import {0}".format(module_name)
                    if j > 1:
                        module_name = '.'.join(pcomponents[:j - 1])
                        base_msg += " but could import {0}".format(module_name)
                    raise ImportError("{0}. Original exception: {1}".format(base_msg, str(e)))
                j += 1
    try:
        obj = eval(tag_suffix)
    except AttributeError as e:
        try:
            pieces = tag_suffix.split('.')
            module = '.'.join(pieces[:-1])
            candidates = dir(eval(module))
            msg = ("Could not evaluate {0}. "
                   "Did you mean {1}? "
                   "Original error was {2}".format(tag_suffix, candidates, str(e)))
        except Exception:
            warnings.warn("Attempt to decipher AttributeError failed")
            raise AttributeError("Could not evaluate {0}. Original error was {1}".format(tag_suffix, str(e)))
        raise AttributeError(msg)
    return obj


def _initialize():
    global IS_INITIALIZED
    yaml.add_multi_constructor("!obj:", _multi_constructor_obj, Loader=yaml.SafeLoader)
    yaml.add_multi_constructor("!import:", _multi_constructor_import, Loader=yaml.SafeLoader)

    yaml.add_constructor("!import", _constructor_import, Loader=yaml.SafeLoader)
    yaml.add_constructor("!float", _constructor_float, Loader=yaml.SafeLoader)

    pattern = re.compile(SCIENTIFIC_NOTATION_REGEXP)
    yaml.add_implicit_resolver("!float", pattern)
    IS_INITIALIZED = True


def _multi_constructor_obj(loader, tag_suffix, node):
    yaml_src = yaml.serialize(node)
    _construct_mapping(node)
    mapping = loader.construct_mapping(node)

    assert hasattr(mapping, 'keys')
    assert hasattr(mapping, 'values')

    for key in mapping.keys():
        if not isinstance(key, six.string_types):
            raise TypeError(
                "Received non string object ({0}) as key in mapping.".format(str(key)))
    if '.' not in tag_suffix:
        my_callable = eval(tag_suffix)
    else:
        my_callable = _try_to_import(tag_suffix)
    rval = _Proxy(callable=my_callable, yaml_src=yaml_src, positionals=(), keywords=mapping)
    return rval


def _multi_constructor_import(loader, tag_suffix, node):
    if '.' not in tag_suffix:
        raise yaml.YAMLError("!import: tag suffix contains no'.'")
    return _try_to_import(tag_suffix)


def _constructor_import(loader, node):
    val = loader.construct_scalar(node)
    if '.' not in val:
        raise yaml.YAMLError("Import tag suffix contains no '.'")
    return _try_to_import(val)


def _constructor_float(loader, node):
    return float(loader.construct_scalar(node))


def _construct_mapping(node):  # , deep=False
    if not isinstance(node, yaml.nodes.MappingNode):
        raise Exception("Expected a mapping node, but found {0} {1}.".format(node.id, node.start_mark))
    mapping = {}
    constructor = yaml.constructor.BaseConstructor()
    for key_node, value_node in node.value:
        key = constructor.construct_object(key_node, deep=False)
        try:
            hash(key)
        except TypeError as exc:
            raise Exception("While constructing a mapping {0}, found unacceptable key ({1})."
                            .format(node.start_mark, (exc, key_node.start_mark)))
        if key in mapping:
            raise Exception("While constructing a mapping {0}, found duplicate key ({1})."
                            .format(node.start_mark, key))
        value = constructor.construct_object(value_node, deep=False)
        mapping[key] = value
    return mapping
