import argparse
import json
import os
import sys
from argparse import Action

try:
    from pyhocon import ConfigFactory, ConfigTree
    import _jsonnet
    import torch
    from torch.distributed.elastic.multiprocessing import Std
    from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
    from torch.distributed.launcher.api import LaunchConfig
except ImportError:
    _jsonnet = None
    ConfigFactory = None

from slickconf.config import Config
from slickconf.container import AnyConfig


def read_config(config_file: str, overrides: tuple = (), strict: bool = False):
    if config_file.endswith(".jsonnet"):
        json_str = _jsonnet.evaluate_file(config_file)
        json_obj = json.loads(json_str)
        conf = ConfigFactory.from_dict(json_obj)

    elif config_file.endswith(".py"):
        from slickconf.builder import PyConfig

        conf = PyConfig.load(config_file)

        return conf

    else:
        conf = ConfigFactory.parse_file(config_file)

    if len(overrides) > 0:
        for override in overrides:
            conf_overrides = ConfigFactory.parse_string(override)
            conf = ConfigTree.merge_configs(conf, conf_overrides)

    return conf.as_plain_ordered_dict()


def add_preset_arguments(parser: argparse.ArgumentParser, elastic: bool = True):
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str)

    if elastic:
        parser = add_elastic_args(parser)

    else:
        parser = add_distributed_args(parser)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    return parser


def preset_argparser(elastic: bool = False):
    parser = argparse.ArgumentParser()

    parser = add_preset_arguments(parser, elastic)

    return parser


def add_distributed_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_machine", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)

    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    return parser


class env(Action):
    def __init__(self, dest, default=None, required=False, **kwargs):
        env_name = f"PET_{dest.upper()}"
        default = os.environ.get(env_name, default)

        if default:
            required = False

        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class check_env(Action):
    def __init__(self, dest, default=None, **kwargs):
        env_name = f"PET_{dest.upper()}"
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))

        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


def parse_min_max_nodes(n_node):
    ar = n_node.split(":")

    if len(ar) == 1:
        min_node = max_node = int(ar[0])

    elif len(ar) == 2:
        min_node, max_node = int(ar[0]), int(ar[1])

    else:
        raise ValueError(f'n_node={n_node} is not in "MIN:MAX" format')

    return min_node, max_node


def local_world_size(n_gpu):
    try:
        return int(n_gpu)

    except ValueError:
        if n_gpu == "cpu":
            n_proc = os.cpu_count()

        elif n_gpu == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")

            n_proc = torch.cuda.device_count()

        elif n_gpu == "auto":
            if torch.cuda.is_available():
                n_proc = torch.cuda.device_count()

            else:
                n_proc = os.cpu_count()

        else:
            raise ValueError(f"Unsupported n_proc value: {n_gpu}")

        return n_proc


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    return port


def get_rdzv_endpoint(args, max_node):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        dist_url = args.dist_url

        if dist_url == "auto":
            if max_node != 1:
                raise ValueError('dist_url="auto" not supported in multi-machine jobs')

            port = find_free_port()
            dist_url = f"127.0.0.1:{port}"

        return dist_url

    return args.rdzv_endpoint


def elastic_config(args):
    min_node, max_node = parse_min_max_nodes(args.n_node)
    n_proc = local_world_size(args.n_proc)

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args, max_node)

    config = LaunchConfig(
        min_nodes=min_node,
        max_nodes=max_node,
        nproc_per_node=n_proc,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        log_dir=args.log_dir,
    )

    return config


def add_elastic_args(parser):
    parser.add_argument("--n_proc", type=str, default="1")
    parser.add_argument("--n_node", type=str, default="1:1")
    parser.add_argument("--node_rank", type=int, default=0)

    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument("--dist_url", default=f"127.0.0.1:{port}")

    parser.add_argument("--rdzv_backend", action=env, type=str, default="static")
    parser.add_argument(
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id", action=env, type=str, default="none", help="User-defined group id."
    )
    parser.add_argument(
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key 1>=<value 1>, <key 2>=<value 2>, ...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend",
    )

    parser.add_argument("--max_restarts", action=env, type=int, default=0)
    parser.add_argument("--monitor_interval", action=env, type=float, default=5)
    parser.add_argument(
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
    )
    parser.add_argument("--role", action=env, type=str, default="default")
    parser.add_argument("--log_dir", action=env, type=str, default=None)
    parser.add_argument("-r", "--redirects", action=env, type=str, default="0")
    parser.add_argument("-t", "--tee", action=env, type=str, default="0")

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


def load_arg_config(
    config_model, show: bool = False, elastic: bool = False, parser=None
):
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
        preset_parser = preset_argparser(elastic=elastic)

    else:
        preset_parser = add_preset_arguments(parser)

    args = preset_parser.parse_args()

    conf = load_config(config_model, args.conf, args.opts, show)
    conf.ckpt = args.ckpt

    if parser is None:
        return conf

    else:
        return conf, args
