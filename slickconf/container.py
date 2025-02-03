from collections.abc import Mapping, Sequence
import copy
import functools
import inspect
from inspect import _ParameterKind
from typing import Any

from torch import normal

try:
    from pydantic_core import core_schema

except ImportError:
    pass

from slickconf.constants import ARGS_KEY, FN_KEY, INIT_KEY, SIGNATURE_KEY


class AnyConfig(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = AnyConfig(**value)
            self[key] = value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e


def unfold_field(x):
    if isinstance(x, Sequence) and not isinstance(x, str):
        return [unfold_field(i) for i in x]

    if isinstance(x, Mapping):
        res = {}

        for k, v in x.items():
            res[k] = unfold_field(v)

        return res

    return x


class Field(dict):
    @classmethod
    def __get_pydantic_core_schema__(self, cls, source_type):
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.dict_schema()
        )

    @classmethod
    def validate(cls, v):
        instance = cls._recursive_init_(**v)

        return instance

    @classmethod
    def _recursive_init_(cls, node):
        if isinstance(node, Sequence) and not isinstance(node, str):
            return [cls._recursive_init_(elem) for elem in node]

        elif isinstance(node, Mapping):
            new_node = cls()

            for key, value in node.items():
                new_node[key] = cls._recursive_init_(value)

            return new_node

        else:
            return node

    def __getitem__(self, item):
        if isinstance(item, int):
            return self["__args"][item]

        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self["__args"][key] = value
        else:
            super().__setitem__(key, value)

    def __getattr__(self, item):
        try:
            return self[item]

        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    def to_dict(self):
        return unfold_field(self)


field = Field


class _SingleCounter:
    counter = 0

    def increase(self, delta=1):
        _SingleCounter.counter += delta


SingleCounter = _SingleCounter()


def get_signature(obj):
    try:
        return inspect.signature(obj)

    except ValueError:
        if isinstance(obj, type) and hasattr(obj, "__call__"):
            try:
                return inspect.signature(obj.__call__)

            except ValueError:
                pass

        raise


class NodeDict(Field):
    @classmethod
    def build(cls, __key, __name, obj, args, kwargs):
        signature = get_signature(obj)
        signature_dict = {}
        for name, param in signature.parameters.items():
            signature_dict[name] = param.kind.value

        arguments = signature.bind_partial(*args, **kwargs).arguments

        for name in list(arguments.keys()):
            kind = signature_dict[name]

            if kind == _ParameterKind.POSITIONAL_ONLY:
                value = arguments.pop(name)

                if ARGS_KEY not in arguments:
                    arguments[ARGS_KEY] = []

                arguments[ARGS_KEY].append(value)

            elif kind == _ParameterKind.VAR_POSITIONAL:
                value = arguments.pop(name)

                if ARGS_KEY not in arguments:
                    arguments[ARGS_KEY] = []

                arguments[ARGS_KEY].extend(value)

            elif kind == _ParameterKind.VAR_KEYWORD:
                arguments.update(arguments.pop(name))

        node = {
            __key: __name,
            SIGNATURE_KEY: signature_dict,
            **arguments,
        }

        return cls(node)

    def _get_max_count(self, node=None):
        if node is None:
            node = self

        max_counter = 0

        for k, v in node.items():
            if k == "__key":
                _, counter = v.rsplit("#", 1)
                max_counter = int(counter)

            elif isinstance(v, Mapping):
                max_counter = max(max_counter, self._get_max_count(v))

        return max_counter

    def __copy__(self):
        new_dict = NodeDict()
        for k, v in self.items():
            if k == "__key":
                obj, _ = v.rsplit("#", 1)
                new_dict[k] = obj + f"#{SingleCounter.counter}"
                SingleCounter.increase()

            else:
                new_dict[k] = copy.deepcopy(v)

        return NodeDict(new_dict)

    def __deepcopy__(self, memo):
        new_dict = NodeDict()
        memo[id(self)] = new_dict

        if "_key_maps_" not in memo:
            memo["_key_maps_"] = {}

        key_maps = memo["_key_maps_"]

        if "__key" in self and self["__key"] in key_maps:
            return key_maps[self["__key"]]

        for k, v in self.items():
            if k == "__key":
                obj, _ = v.rsplit("#", 1)
                v = obj + f"#{SingleCounter.counter}"
                SingleCounter.increase()

            new_dict[copy.deepcopy(k, memo)] = copy.deepcopy(v, memo)

        if "__key" in self:
            key_maps[self["__key"]] = new_dict

        return NodeDict(new_dict)

    def __repr__(self):
        return dict.__repr__(self)


class NodeDictProxyObject(dict):
    def __init__(
        self,
        name: str,
        parent: Any | None = None,
        is_import: bool = False,
    ):
        self.name = name
        self.parent = parent
        self.is_import = is_import
        self.target = None

        self.refresh()

    @classmethod
    @functools.cache
    def from_cache(cls, **kwargs):
        return cls(**kwargs)

    @property
    def qualname(self):
        if not self.parent:
            return self.name

        if self.parent.is_import and not self.is_import:
            separator = ":"

        else:
            separator = "."

        return f"{self.parent.qualname}{separator}{self.name}"

    def refresh(self):
        super().__init__({FN_KEY: self.qualname})

    def __getattr__(self, name):
        obj = type(self).from_cache(name=name, parent=self)

        if self.target is not None:
            obj.set_target(getattr(self.target, name))

        obj.refresh()

        return obj

    def set_target(self, obj):
        self.target = obj

        try:
            signature = get_signature(obj)

        except (ValueError, TypeError) as _:
            return

        signature_dict = {}
        for name, param in signature.parameters.items():
            signature_dict[name] = param.kind.value

        self[SIGNATURE_KEY] = signature_dict

    def child_import(self, name: str, target: Any | None = None):
        obj = getattr(self, name)
        obj.is_import = True

        if target is not None:
            obj.set_target(target)

        obj.refresh()

        return obj

    def __call__(self, *args, **kwargs):
        if self.target is not None:
            return NodeDict.build(INIT_KEY, self.qualname, self.target, args, kwargs)

        return NodeDict(
            {
                INIT_KEY: self.qualname,
                ARGS_KEY: list(args),
                **kwargs,
            }
        )

    def __bool__(self):
        return True

    __eq__ = object.__eq__
    __hash__ = object.__hash__
