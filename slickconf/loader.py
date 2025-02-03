import argparse
import ast
import copy
import collections
from collections.abc import Mapping, Sequence
import json
import os
import re
import sys

try:
    from pyhocon import ConfigFactory, ConfigTree
    import _jsonnet

except ImportError:
    _jsonnet = None
    ConfigFactory = None

from pydantic import BaseModel

from slickconf.config import Config
from slickconf.container import Field, NodeDict, SingleCounter
from slickconf.constants import KEY_KEY
from slickconf.tree import traverse


import ast
import re


def parse_expr(expr):
    left, right = expr.split("=")
    left = left.strip()

    try:
        right = ast.literal_eval(right)

    except (SyntaxError, ValueError) as _:
        pass

    parts = left.split(".")

    return parts, right


def merge_dict(left, right):
    new_dict = {}

    model_class = None
    if isinstance(left, BaseModel):
        model_class = left.__class__
        left = left.model_dump()

    for key, val in left.items():
        if isinstance(val, collections.abc.Mapping):
            new_dict[key] = copy.copy(val)

        else:
            new_dict[key] = val

    for key, val in right.items():
        if key in new_dict:
            if isinstance(
                new_dict[key], (collections.abc.Mapping, BaseModel)
            ) and isinstance(val, (collections.abc.Mapping, BaseModel)):
                new_dict[key] = merge_dict(new_dict[key], val)

            else:
                new_dict[key] = right[key]

        else:
            new_dict[key] = val

    if model_class is not None:
        new_dict = model_class(**new_dict)

    return new_dict


def apply_expr(target, parts, val):
    for i, part in enumerate(parts):
        match = re.match(r"(.*?)\[(\d+)\]", part)

        if match is not None:
            key, id = match.groups()
            id = ast.literal_eval(id)

            if i == len(parts) - 1:
                getattr(target, key)[id] = val

            else:
                target = getattr(target, key)[id]

        else:
            key = part

            if isinstance(target, dict):
                if i == len(parts) - 1:
                    target[key] = val

                else:
                    target = target[key]

            else:
                if i == len(parts) - 1:
                    setattr(target, key, val)

                else:
                    target = getattr(target, key)


def apply_overrides(target, overrides):
    for override in overrides:
        apply_expr(target, *parse_expr(override))

    return target


def read_config(config_file: str, overrides: tuple = (), config_name: str = "conf"):
    if config_file.endswith(".jsonnet"):
        json_str = _jsonnet.evaluate_file(config_file)
        json_obj = json.loads(json_str)
        conf = ConfigFactory.from_dict(json_obj)

    elif config_file.endswith(".py"):
        from slickconf.builder import PyConfig

        conf = PyConfig.load(config_file, config_name=config_name)

        for override in overrides:
            apply_expr(conf, *parse_expr(override))

        return conf

    elif config_file.endswith(".toml"):
        try:
            import tomllib

        except ImportError:
            try:
                import tomli as tomllib

            except ImportError:
                try:
                    import tomlkit as tomllib

                except ImportError:
                    raise ImportError(
                        "at least one of tomllib, tomli, or tomlkit is required"
                    )

        with open(config_file, "rb") as f:
            conf = tomllib.load(f)

        return conf

    else:
        conf = ConfigFactory.parse_file(config_file)

    if len(overrides) > 0:
        for override in overrides:
            conf_overrides = ConfigFactory.parse_string(override)
            conf = ConfigTree.merge_configs(conf, conf_overrides)

    return conf.as_plain_ordered_dict()


# Supports multiple config files as an argument, but when single config file is provided, return it instead of a list
class ConfAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            setattr(namespace, self.dest, values[0])

        else:
            setattr(namespace, self.dest, values)


def add_preset_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--conf_name", type=str, default="conf")
    parser.add_argument("--ckpt", type=str)

    parser = add_distributed_args(parser)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    return parser


def preset_argparser():
    parser = argparse.ArgumentParser()

    parser = add_preset_arguments(parser)

    return parser


def add_distributed_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_machine", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)

    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    return parser


def build_config(overrides: tuple, config_model: Config | None = None):
    field = Field()

    for override in overrides:
        apply_expr(field, *parse_expr(override))

    if config_model is None:
        return field

    return config_model(**field)


def load_task_config(config: str, task: str, overrides: tuple = ()):
    task = read_config(config)[task]
    task_conf = {}

    for key, val in task.items():
        if isinstance(val, str):
            task_conf[key] = read_config(val)

        elif isinstance(val, list):
            task_conf[key] = []

            for fname in val:
                task_conf[key].append(read_config(fname))

        else:
            task_conf[key] = val

    return task_conf


def load_config(
    config: str,
    config_model: Config | None = None,
    config_name: str = "conf",
    overrides_file=None,
    overrides: tuple = (),
):
    """
    Loads the configuration from a given source and applies any overrides.

    This function reads the configuration from the specified source, applies any provided overrides, and then
    validates the configuration against an optional configuration model. If no model is provided, a generic
    configuration model is used. The final, validated configuration is then returned.

    Parameters:
    - config: The source of the configuration to be loaded.
    - config_model: An optional Pydantic model class that the loaded configuration will be validated against. If not
                    provided, a AnyConfig that accepts any configuration is used.
    - overrides: An optional tuple of overrides that will be applied to the configuration. Each override is a
                 key-value pair where the key specifies the configuration setting to be overridden and the value
                 specifies the new value for that setting.

    Returns:
    - The loaded and validated configuration as an instance of the specified `config_model` or a AnyConfig
      object if no model was specified.
    """
    conf = read_config(config, config_name=config_name)

    if overrides_file is not None:
        overrides_file = read_config(overrides_file)
        conf = merge_dict(conf, overrides_file["overrides"])

    conf = apply_overrides(conf, overrides)

    if config_model is None:
        return NodeDict._recursive_init_(conf)

    return config_model(**conf)


def load_arg_config(config_model, show: bool = False, parser=None):
    """
    Loads configuration based on command-line arguments and applies overrides.

    This function initializes the argument parser with or without elastic specific arguments based on the `elastic` flag.
    It then parses the command-line arguments, loads the configuration from the specified source, applies any provided
    overrides, and optionally displays the configuration. The final, validated configuration is then returned along with
    the parsed arguments if the parser is not provided externally.

    Parameters:
    - config_model: The Pydantic model class that the loaded configuration will be validated against.
    - show: A boolean flag indicating whether to print the loaded configuration to stdout.
    - elastic: A boolean flag indicating whether to initialize the argument parser with elastic specific arguments.
    - parser: An optional external argument parser. If not provided, a new parser will be initialized based on the `elastic` flag.

    Returns:
    - The loaded and validated configuration as an instance of the specified `config_model`.
    - The parsed arguments, if the parser is not provided externally.
    """

    if parser is None:
        preset_parser = preset_argparser()

    else:
        preset_parser = add_preset_arguments(parser)

    args = preset_parser.parse_args()

    conf = load_config(
        args.conf, config_model, config_name=args.conf_name, overrides=args.opts
    )
    conf.ckpt = args.ckpt

    if parser is None:
        return conf

    else:
        return conf, args


def _get_min_max_single_count(root):
    min_count = float("inf")
    max_count = 0

    for node in traverse(root):
        if KEY_KEY in node:
            _, counter = node[KEY_KEY].rsplit("#", 1)
            min_count = min(min_count, int(counter))
            max_count = max(max_count, int(counter))

    return min_count, max_count


def deserialize(path_or_conf: str | Mapping):
    conf = path_or_conf

    if isinstance(path_or_conf, str):
        with open(path_or_conf, "r") as f:
            conf = json.load(f)

    min_count, max_count = _get_min_max_single_count(conf)

    for node in traverse(conf):
        if KEY_KEY in node:
            obj, counter = node[KEY_KEY].rsplit("#", 1)
            counter = int(counter)
            counter = counter - min_count + SingleCounter.counter
            node[KEY_KEY] = f"{obj}#{counter}"

    SingleCounter.increase(max_count - min_count + 1)

    return NodeDict._recursive_init_(conf)
