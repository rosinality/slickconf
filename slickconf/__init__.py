from slickconf.builder import (
    F,
    L,
    placeholder,
    single,
    function,
    tag,
    call,
    config_fn,
    annotate,
    get_instance_attr,
    exempt,
    repeat,
)
from slickconf.container import Field, field, NodeDict
from slickconf.config import Config, Instance, MainConfig, instantiate
from slickconf.constants import DEFAULT
from slickconf.imports import imports
from slickconf.loader import build_config, load_arg_config, load_config, deserialize
from slickconf.patch import patch
from slickconf.selector import select
from slickconf.summarize import summarize
from slickconf.tree import (
    _register_pytree_node,
    _dict_flatten,
    _dict_flatten_with_keys,
)


def _field_unflatten(values, context):
    return Field(zip(context, values))


def _nodedict_unflatten(values, context):
    return NodeDict(zip(context, values))


def _instance_unflatten(values, context):
    return Instance(zip(context, values))


_register_pytree_node(
    Field,
    _dict_flatten,
    _field_unflatten,
    flatten_with_keys_fn=_dict_flatten_with_keys,
)

_register_pytree_node(
    NodeDict,
    _dict_flatten,
    _nodedict_unflatten,
    flatten_with_keys_fn=_dict_flatten_with_keys,
)

_register_pytree_node(
    Instance,
    _dict_flatten,
    _instance_unflatten,
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
