import collections
from copy import deepcopy

from slickconf.config import instantiate


def run_hook(node, hooks):
    node = deepcopy(node)
    _run_hook(node, hooks)

    return node


def _run_hook(node, hooks):
    target_key = "__target"
    init_key = "__init"
    fn_key = "__fn"
    validate_key = "__validate"
    partial_key = "__partial"
    args_key = "__args"
    key_key = "__key"

    exclude_keys = {
        target_key,
        init_key,
        fn_key,
        validate_key,
        partial_key,
        args_key,
        key_key,
    }

    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        seq = [_run_hook(n, hooks) for i, n in enumerate(node)]

        return seq

    if isinstance(node, collections.abc.Mapping):
        if target_key in node or init_key in node:
            if init_key in node:
                target = node.get(init_key)
                fn_key = init_key

            else:
                target = node.get(target_key)
                fn_key = target_key

            for config_key, config_hook in hooks.items():
                if target == config_key:
                    if isinstance(config_hook, str):
                        node[fn_key] = config_hook
                        res = instantiate(node, conf=node)
                        node.clear()
                        node.update(res)

                    else:
                        new_node = config_hook(node)
                        node.clear()
                        node.update(new_node)

            if args_key in node:
                args_node = node[args_key]

                for arg in args_node:
                    _run_hook(arg, hooks)

            for k, v in node.items():
                if k in exclude_keys:
                    continue

                _run_hook(v, hooks)

    return node
