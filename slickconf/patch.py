import collections
import itertools

from slickconf.selector import select
from slickconf.container import NodeDict
from slickconf.constants import INIT_KEY, SIGNATURE_KEY, ARGS_KEY


def _setattr(tree, path, value):
    paths = path.split(".")

    for path in paths[:-1]:
        tree = tree[path]

    tree[paths[-1]] = value

    return tree


def _select(tree):
    return select(tree)


def _at(selection, key):
    return selection.at(lambda x: x[key])


def _instance(selection, instance):
    return selection.instance(instance)


def _set_sequence(selection, replace, start=0, step=1):
    return selection.set_sequence(
        (replace_params(replace, index=index) for index in itertools.count(start, step))
    )


def _update_dict(selection, value):
    return selection.update_dict(value)


def replace_params(param_str, **kwargs):
    if not isinstance(param_str, str):
        return param_str

    if param_str.startswith("$"):
        param_str = param_str[1:]

        try:
            param = kwargs[param_str]

        except KeyError:
            raise KeyError(f"Parameter {param_str} not found in kwargs")

        return param

    param_str.format(**kwargs)


def _map_instance(selection, target, signature, args, kwargs):
    def map_instance(x):
        new_args = []
        for arg in args:
            try:
                new_args.append(replace_params(arg, **x))

            except KeyError:
                continue

        new_kwargs = {}
        for k, v in kwargs.items():
            try:
                new_kwargs[k] = replace_params(v, **x)

            except KeyError:
                continue

        new_dict = {INIT_KEY: target, **new_kwargs}

        if len(new_args) > 0:
            new_dict[ARGS_KEY] = new_args

        if signature is not None:
            new_dict[SIGNATURE_KEY] = signature

        return NodeDict(new_dict)

    return selection.map(map_instance)


def _do_patch(tree, step):
    type = step["type"]

    if type == "setattr":
        tree = _setattr(tree, step["path"], step["value"])

    elif type == "select":
        tree = _select(tree)

    elif type == "instance":
        tree = _instance(tree, step["instance"])

    elif type == "at":
        tree = _at(tree, step["key"])

    elif type == "update_dict":
        tree = _update_dict(tree, step["value"])

    elif type == "map_instance":
        tree = _map_instance(
            tree,
            step["target"],
            step.get("signature", None),
            step.get("args", ()),
            step.get("kwargs", {}),
        )

    elif type == "set_sequence":
        tree = _set_sequence(
            tree, step["replace"], step.get("start", 0), step.get("step", 1)
        )

    return tree


def patch(tree, patches):
    for patch in patches:
        if isinstance(patch, collections.abc.Sequence):
            for sub_patch in patch:
                tree = _do_patch(tree, sub_patch)
        else:
            tree = _do_patch(tree, patch)

    return tree
