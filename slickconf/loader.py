import argparse
import ast
import json
import os
import re
import sys
from argparse import Action

try:
    from pyhocon import ConfigFactory, ConfigTree
    import _jsonnet
    import torch

except ImportError:
    _jsonnet = None
    ConfigFactory = None

from slickconf.config import Config
from slickconf.container import AnyConfig


import ast
import re


def parse_expr(expr):
    left, right = expr.split("=")
    left = left.strip()
    right = ast.literal_eval(right)
    parts = left.split(".")

    return parts, right


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


def read_config(config_file: str, overrides: tuple = ()):
    if config_file.endswith(".jsonnet"):
        json_str = _jsonnet.evaluate_file(config_file)
        json_obj = json.loads(json_str)
        conf = ConfigFactory.from_dict(json_obj)

    elif config_file.endswith(".py"):
        from slickconf.builder import PyConfig

        conf = PyConfig.load(config_file)

        for override in overrides:
            apply_expr(conf, *parse_expr(override))

        return conf

    else:
        conf = ConfigFactory.parse_file(config_file)

    if len(overrides) > 0:
        for override in overrides:
            conf_overrides = ConfigFactory.parse_string(override)
            conf = ConfigTree.merge_configs(conf, conf_overrides)

    return conf.as_plain_ordered_dict()


def add_preset_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--conf", type=str, required=True)
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


def load_config(config: str, config_model: Config | None = None, overrides: tuple = ()):
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

    conf = read_config(config, overrides=overrides)

    if config_model is None:
        config_model = AnyConfig

    conf = config_model(**conf)

    return conf


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

    conf = load_config(args.conf, config_model, args.opts)
    conf.ckpt = args.ckpt

    if parser is None:
        return conf

    else:
        return conf, args
