import collections

from pydantic import BaseModel

from slickconf.config import Instance
from slickconf.constants import (
    INIT_KEY,
    FN_KEY,
    TARGET_KEY,
    KEY_KEY,
    ARGS_KEY,
    EXCLUDE_KEYS,
    ANNOTATE_KEY,
)


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class ListFormatter(list):
    def __repr__(self):
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return "[]"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1

                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        if all([start == end for start, end in start_end_indices]):
            return super().__repr__()

        lines = []
        main_str = "["
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += "]"
        return main_str


class CallFormatter:
    def __init__(self, name, args, kwargs, key, partial):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.key = key
        self.partial = partial

        if self.key is not None:
            self.key = self.key.rsplit("#", 1)[-1]

    def format_val(self, val):
        if isinstance(val, str):
            val = f'"{val}"'

        else:
            val = str(val)

        return _addindent(val, 2)

    def __repr__(self):
        args = [self.format_val(v) for v in self.args]
        kwargs = [f"{k}={self.format_val(v)}" for k, v in self.kwargs.items()]
        MAX_LEN = 79

        params = args + kwargs

        if self.partial:
            params = [self.name] + params

        if len(", ".join(params)) < MAX_LEN:
            inline = True
            params = ", ".join(params)

        else:
            inline = False
            params = ",\n  ".join(params)

        if self.partial:
            if inline:
                result = f"partial({params})"

            else:
                result = f"partial(\n  {params}\n)"

        else:
            if inline:
                result = f"{self.name}({params})"

            else:
                result = f"{self.name}(\n  {params}\n)"

        return result


class AnnotateFormatter:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"[{self.key}]"


def patch_dict(node):
    partial = False
    if INIT_KEY in node:
        target = node[INIT_KEY]

    elif FN_KEY in node:
        target = node[FN_KEY]
        partial = True

    elif TARGET_KEY in node:
        target = node[TARGET_KEY]

    elif ANNOTATE_KEY in node:
        return AnnotateFormatter(node[ANNOTATE_KEY])

    else:
        return node

    args = []
    if ARGS_KEY in node:
        args = node[ARGS_KEY]

    kwargs = {}
    for k, v in node.items():
        if k in EXCLUDE_KEYS:
            continue

        kwargs[k] = v

    key = None
    if KEY_KEY in node:
        key = node[KEY_KEY]

    return CallFormatter(target, args, kwargs, key, partial)


def summarize(node, skip_keys=()):
    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        return ListFormatter([summarize(n) for n in node])

    elif isinstance(node, (collections.abc.Mapping, BaseModel, Instance)):
        results = {}

        node_dict = node
        if isinstance(node, BaseModel):
            node_dict = node_dict.dict()

        for k, v in node_dict.items():
            if k in skip_keys:
                continue

            results[k] = summarize(v)

        results = patch_dict(results)

        if isinstance(node, BaseModel):
            results = node.construct(**results)

        elif isinstance(node, Instance):
            results = Instance(**results)

        return results

    else:
        return node
