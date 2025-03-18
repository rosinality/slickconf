import json
from typing import Callable

from pydantic import StrictInt


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(repr, self.layers))})"


class Container:
    def __init__(self, value):
        self.value = value


class NamedContainer:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class BetterContainer:
    def __init__(self, name, value):
        self.name = name
        self.value = value


def nested_dict(id: int, name: str, fields: dict[str, Callable]):
    return id, name, fields


def blank_fn(input):
    return input


def simple_fn(arg1, arg2, arg3):
    return " ".join((arg1, arg2, arg3))


def is_serializable(obj):
    try:
        json.dumps(obj)

        return True

    except Exception:
        return False


def fn_args(a, *args, b):
    return a, args, b


def fn_plain(a, b):
    return a, b


def fn_position_only(a, /, b):
    return a, b


def fn_keyword_only(a, *, b):
    return a, b


def fn_kwargs(a, **kwargs):
    return a, tuple(kwargs.items())


def typed_fn(a: int, b: StrictInt) -> int:
    return a + b
