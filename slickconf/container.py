from collections.abc import Mapping, Sequence
from collections import namedtuple
import copy
import functools
import inspect
from inspect import _ParameterKind
import operator
from typing import Any

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


# from ml_collections
# https://github.com/google/ml_collections/blob/master/ml_collections/config_dict/config_dict.py

_NoneType = type(None)


def _is_callable_type(field_type):
    """Tries to ensure: `_is_callable_type(type(obj)) == callable(obj)`."""
    return any("__call__" in c.__dict__ for c in field_type.__mro__)


def _is_type_safety_violation(value, field_type):
    """Helper function for type safety exceptions.

    This function determines whether or not assigning a value to a field violates
    type safety.

    Args:
      value: The value to be assigned.
      field_type: Type of the field that we would like to assign value to.

    Returns:
      True if assigning value to field violates type safety, False otherwise.
    """
    # Allow None to override and be overridden by any type.
    if value is None or field_type == _NoneType:
        return False
    elif isinstance(value, field_type):
        return False
    else:
        # A callable can overridde a callable.
        return not (callable(value) and _is_callable_type(field_type))


def _safe_cast(value, field_type, type_safe=False):
    """Helper function to handle the exceptional type conversions.

    This function implements the following exceptions for type-checking rules:

    * An `int` will be converted to a `float` if overriding a `float` field.
    * Any string value can override a `str` or `unicode` field. The value is
    converted to `field_type`.
    * A `tuple` will be converted to a `list` if overriding a `list` field.
    * A `list` will be converted to a `tuple` if overriding `tuple` field.
    * Short and long integers are indistinguishable. The final value will always
    be a `long` if both types are present.

    Args:
      value: The value to be assigned.
      field_type: The type for the field that we would like to assign value to.
      type_safe: If True, the method will throw an error if the `value` is not of
          type `field_type` after safe type conversions.

    Returns:
      The converted type-safe version of the value if it is one of the cases
      described. Otherwise, return the value without conversion.

    Raises:
      TypeError: if types don't match  after safe type conversions.
    """
    original_value_type = type(value)

    # The int->float exception.
    if isinstance(value, int) and field_type is float:
        return float(value)

    # The unicode/string to string exception.
    if isinstance(value, str) and field_type is str:
        return field_type(value)

    # tuple<->list conversion. JSON serialization converts lists to tuples, so
    # we need this to avoid errors when overriding a list field with its
    # deserialized version. See b/34805906 for more details.
    if isinstance(value, tuple) and field_type is list:
        return list(value)
    if isinstance(value, list) and field_type is tuple:
        return tuple(value)

    if isinstance(value, int) and field_type is int:
        return value

    if type_safe and _is_type_safety_violation(value, field_type):
        raise TypeError(
            "{} is of original type {} and cannot be casted to type {}".format(
                value, str(original_value_type), str(field_type)
            )
        )
    return value


class _Op(namedtuple("_Op", ["fn", "args"])):
    """A named tuple representing a lazily computed op.

    The _Op named tuple has two fields:
      fn: The function to be applied.
      args: a tuple/list of arguments that are used with the op.
    """


def _get_computed_value(value_or_fieldreference):
    if isinstance(value_or_fieldreference, FieldReference):
        return value_or_fieldreference.get()
    return value_or_fieldreference


class FieldReference:
    def __init__(
        self,
        default: Any,
        field_type: Any | None = None,
        op: Any | None = None,
        required: bool = False,
    ):
        self._value = None

        if field_type is None:
            if default is None:
                raise ValueError("default value cannot be None if field_type is None")

            elif isinstance(default, FieldReference):
                field_type = default.get_type()

            else:
                field_type = type(default)

        else:
            try:
                isinstance(None, field_type)

            except TypeError:
                raise TypeError(f"field_type should be a type, not {type(field_type)}")

        self._field_type = field_type
        self.set(default)

        if required and op is not None:
            raise ValueError("cannot set required to True if op is not None")

        self._required = required
        self._ops = [] if op is None else [op]

    __hash__ = None

    def has_cycle(self, visited=None):
        visited = visited or set()

        if id(self) in visited:
            return True

        visited.add(id(self))

        value = self._value
        if isinstance(value, FieldReference) and value.has_cycle(visited.copy()):
            return True

        for op in self._ops:
            for arg in op.args:
                if isinstance(arg, FieldReference) and arg.has_cycle(visited.copy()):
                    return True

        return False

    def set(self, value, type_safe=True):
        self._ops = []

        if value is None:
            self._value = None

        elif isinstance(value, FieldReference):
            if type_safe and not issubclass(value.get_type(), self.get_type()):
                raise TypeError(
                    f"reference is of type {value.get_type()}, expected {self.get_type()}"
                )

            old_value = getattr(self, "_value", None)
            self._value = value

            if self.has_cycle():
                self._value = old_value

                raise RuntimeError("cycle detected")

        else:
            self._value = _safe_cast(value, self._field_type, type_safe)

    def get_type(self):
        return self._field_type

    def __getattr__(self, name: str):
        return self._apply_op(operator.attrgetter(name), new_type=object)

    def __call__(self, *args, **kwargs):
        n_args = len(args)
        kwargs_keys = list(kwargs.keys())

        def flow_fn(value, *flat_args):
            final_args = flat_args[:n_args]
            final_kwargs = dict(zip(kwargs_keys, flat_args[n_args:]))

            return value(*final_args, **final_kwargs)

        return self._apply_op(flow_fn, *args, **kwargs.values(), new_type=object)

    def _apply_op(self, fn, *args, new_type: Any = None):
        args = [_safe_cast(arg, self._field_type) for arg in args]

        if new_type is None:
            new_type = self._field_type

        return FieldReference(self, field_type=new_type, op=_Op(fn, args))

    def get(self):
        if self._required and self._value is None:
            raise ValueError("required field is not set")

        value = _get_computed_value(self._value)

        for op in self._ops:
            args = [_get_computed_value(arg) for arg in op.args]
            value = op.fn(value, *args)
            value = _get_computed_value(value)

        return value

    def __add__(self, other):
        return self._apply_op(operator.add, other)

    def __radd__(self, other):
        radd = functools.partial(operator.add, other)

        return self._apply_op(radd)

    def __sub__(self, other):
        return self._apply_op(operator.sub, other)

    def __rsub__(self, other):
        rsub = functools.partial(operator.sub, other)

        return self._apply_op(rsub)

    def __mul__(self, other):
        return self._apply_op(operator.mul, other)

    def __rmul__(self, other):
        rmul = functools.partial(operator.mul, other)

        return self._apply_op(rmul)

    def __div__(self, other):
        return self._apply_op(operator.truediv, other)

    def __rdiv__(self, other):
        rdiv = functools.partial(operator.truediv, other)

        return self._apply_op(rdiv)

    def __truediv__(self, other):
        return self._apply_op(operator.truediv, other)

    def __rtruediv__(self, other):
        rtruediv = functools.partial(operator.truediv, other)

        return self._apply_op(rtruediv)

    def __floordiv__(self, other):
        return self._apply_op(operator.floordiv, other)

    def __rfloordiv__(self, other):
        rfloordiv = functools.partial(operator.floordiv, other)

        return self._apply_op(rfloordiv)

    def __pow__(self, other):
        return self._apply_op(operator.pow, other)

    def __mod__(self, other):
        return self._apply_op(operator.mod, other)

    def __and__(self, other):
        return self._apply_op(operator.and_, other)

    def __or__(self, other):
        return self._apply_op(operator.or_, other)

    def __xor__(self, other):
        return self._apply_op(operator.xor, other)

    def __neg__(self):
        return self._apply_op(operator.neg)

    def __abs__(self):
        return self._apply_op(operator.abs)

    def to_int(self):
        return self._apply_op(int)

    def to_float(self):
        return self._apply_op(float)

    def to_str(self):
        return self._apply_op(str)


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

    ref = property(FieldReference)


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
        if isinstance(obj, NodeDictProxyObject):
            obj = obj.target

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
            node = NodeDict.build(INIT_KEY, self.qualname, self.target, args, kwargs)

        else:
            node = NodeDict(
                {
                    INIT_KEY: self.qualname,
                    ARGS_KEY: list(args),
                    **kwargs,
                }
            )

        key = f"{self.qualname}#{SingleCounter.counter}"
        SingleCounter.increase()
        node["__key"] = key

        return node

    def __bool__(self):
        return True

    __eq__ = object.__eq__
    __hash__ = object.__hash__
