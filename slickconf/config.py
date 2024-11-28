import collections
import functools
import inspect
import os
import sys
import typing
from typing import Any, Callable, Optional

from pydantic import (
    BaseModel,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
    validator,
)

try:
    from pydantic_core import core_schema

except ImportError:
    pass

CONFIG_REGISTRY = {}
SINGLETON = {}


class Config(BaseModel):
    class Config:
        extra = "forbid"
        protected_namespaces = ()


class StrictConfig:
    extra = "forbid"
    arbitrary_types_allowed = True


class AnyConfig:
    arbitrary_types_allowed = True


class MainConfig(BaseModel):
    class Config:
        extra = "forbid"

    distributed: Optional[StrictBool] = None
    host: Optional[StrictStr] = None
    world_size: Optional[StrictInt] = None
    rank: Optional[StrictInt] = None
    local_rank: Optional[StrictInt] = None
    ckpt: Optional[StrictStr] = None


def _check_type(type_name):
    @validator("type", allow_reuse=True)
    def check_type(cls, v):
        if v != type_name:
            raise ValueError(f"Type does not match for {type_name}")

        return v

    return check_type


def make_model_from_signature(
    name: str,
    init_fn: Callable[..., Any],
    signature: inspect.Signature,
    exclude: tuple | list,
    type_name: str | None = None,
    strict: bool = True,
):
    """Create a Pydantic model from a function signature."""

    params = {}

    if type_name is not None:
        params["type"] = (StrictStr, ...)

    for k, v in signature.parameters.items():
        if k in exclude:
            continue

        if v.kind == v.VAR_POSITIONAL or v.kind == v.VAR_KEYWORD:
            strict = False

            continue

        annotation = v.annotation
        if annotation is inspect._empty:
            annotation = typing.Any

        if v.default is inspect._empty:
            params[k] = (annotation, ...)

        else:
            params[k] = (annotation, v.default)

    def _params(self):
        values = self.dict()

        if type_name is not None:
            values.pop("type")

        return values

    @functools.wraps(init_fn)
    def _init_fn(self, *args, **kwargs):
        params = self.params()
        params.update(kwargs)
        pos_replace = list(signature.parameters.keys())[: len(args)]
        for pos in pos_replace:
            params.pop(pos)

        return init_fn(*args, **params)

    validators = {"params": _params, "make": _init_fn}

    if type_name is not None:
        validators["check_type"] = _check_type(type_name)

    if strict:
        config = StrictConfig

    else:
        config = AnyConfig

    model = create_model(
        name,
        __config__=config,
        __validators__=validators,
        __module__=__name__,
        **params,
    )

    setattr(sys.modules[__name__], name, model)

    return model


def resolve_module(path: str):
    """
    Resolves a module or attribute within a module from a dotted path.

    This function attempts to import a module or attribute using a dotted path string. It starts by trying to import the
    entire path as a module. If that fails, it progressively steps back through the path, attempting to import each
    segment as a module until it finds a valid module. Then, it attempts to resolve any remaining path segments as
    attributes within the found module.

    Parameters:
    - path (str): The dotted path string to resolve.

    Returns:
    - The resolved module or attribute.

    Raises:
    - ImportError: If the module or attribute cannot be found or imported.
    """

    from importlib import import_module

    sub_path = path.split(".")
    module = None

    for i in reversed(range(len(sub_path))):
        try:
            mod = ".".join(sub_path[:i])
            module = import_module(mod)

        except (ModuleNotFoundError, ImportError):
            continue

        if module is not None:
            break

    obj = module

    for sub in sub_path[i:]:
        mod = f"{mod}.{sub}"

        if not hasattr(obj, sub):
            try:
                import_module(mod)

            except (ModuleNotFoundError, ImportError) as e:
                raise ImportError(
                    f"Encountered error: '{e}' when loading module '{path}'"
                ) from e

        obj = getattr(obj, sub)

    return obj


def resolve_module_pyfile(path: str, filepath):
    import importlib

    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(filepath))[0], filepath
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, path)

    return function


def flatten_tree(node):
    res = []

    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        for n in node:
            res.extend(flatten_tree(n))

        return res

    if isinstance(node, collections.abc.Mapping):
        res.append(node)

        for v in node.values():
            res.extend(flatten_tree(v))

    return res


def find_placeholder(node):
    target_key = "__target"
    init_key = "__init"
    fn_key = "__fn"
    validate_key = "__validate"
    partial_key = "__partial"
    args_key = "__args"
    key_key = "__key"
    placeholder_key = "__placeholder"

    exclude_keys = {
        target_key,
        init_key,
        fn_key,
        validate_key,
        partial_key,
        args_key,
        key_key,
    }

    placeholders = set()

    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        seq = [find_placeholder(n) for i, n in enumerate(node)]

        for s in seq:
            placeholders = placeholders.union(s)

        return placeholders

    if isinstance(node, collections.abc.Mapping):
        if target_key in node or init_key in node or fn_key in node:
            if args_key in node:
                args_node = node[args_key]

                for arg in args_node:
                    placeholders = placeholders.union(find_placeholder(arg))

            for k, v in node.items():
                if k in exclude_keys:
                    continue

                try:
                    if v[placeholder_key] == placeholder_key:
                        placeholders.add(k)

                        continue

                except:
                    pass

                placeholders = placeholders.union(find_placeholder(v))

    return placeholders


def _instance_traverse_validate(
    node,
    recursive=True,
    instantiate=False,
    _tags_=None,
):
    target_key = "__target"
    init_key = "__init"
    fn_key = "__fn"
    validate_key = "__validate"
    partial_key = "__partial"
    args_key = "__args"
    key_key = "__key"
    tag_key = "__tag"

    exclude_keys = {
        target_key,
        init_key,
        fn_key,
        validate_key,
        partial_key,
        args_key,
        key_key,
        "__meta",
    }

    partial = node.get(partial_key, False)
    do_validate = node.get(validate_key, True)

    if init_key in node:
        target = node.get(init_key)

    elif fn_key in node:
        target = node.get(fn_key)

        if len([k for k in node if k not in exclude_keys]) > 0:
            partial = True

        else:
            do_validate = False

    else:
        target = node.get(target_key)

    if node.get("__meta", None) is not None and node["__meta"].get(
        "import_pyfile", False
    ):
        obj = resolve_module_pyfile(
            node["__meta"]["qualname"], node["__meta"]["filepath"]
        )

    else:
        obj = resolve_module(target)

    signature = inspect.signature(obj)

    rest = {}

    args_replaced = []
    if args_key in node:
        for arg, k in zip(node[args_key], signature.parameters.keys()):
            rest[k] = arg
            args_replaced.append(k)

    for k, v in node.items():
        if k in exclude_keys:
            continue

        rest[k] = instance_traverse(
            v, recursive=recursive, _tags_=_tags_, instantiate=instantiate
        )

        if k in args_replaced:
            raise TypeError(f"{target} got multiple values for argument '{k}'")

    if do_validate:
        name = "instance." + target

        exclude = []

        for r_k, r_v in rest.items():
            for e_k in exclude_keys:
                try:
                    if not isinstance(r_v, str) and e_k in r_v:
                        exclude.append(r_k)

                        break

                except:
                    continue

            try:
                if not isinstance(r_v, str) and tag_key in r_v:
                    exclude.append(r_k)

            except:
                pass

        if partial:
            rest_key = list(rest.keys())

            for k in signature.parameters.keys():
                if k not in rest_key:
                    exclude.append(k)

            model = make_model_from_signature(
                name, obj, signature, exclude, strict=False
            )

        else:
            model = make_model_from_signature(name, obj, signature, exclude)

        try:
            if len(exclude) > 0:
                exclude = set(exclude)
                model.validate({k: v for k, v in rest.items() if k not in exclude})

            else:
                model.validate(rest)

        except ValidationError as e:
            arbitrary_flag = True

            for error in e.errors():
                if error["type"] != "type_error.arbitrary_type":
                    arbitrary_flag = False

                    break

            if not arbitrary_flag:
                raise ValueError(
                    f"Validation for {target} with {v} is failed:\n{e}"
                ) from e

    for arg in args_replaced:
        del rest[arg]

    return_dict = {**node, **rest}

    return return_dict


def _instance_traverse_instantiate(
    node,
    *args,
    recursive=True,
    instantiate=False,
    keyword_args=None,
    _tags_=None,
    root=True,
    singleton_dict=None,
):
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
        "__meta",
    }

    return_fn = False
    partial = node.get(partial_key, False)

    if init_key in node:
        target = node.get(init_key)

    elif fn_key in node:
        target = node.get(fn_key)

        if len([k for k in node if k not in exclude_keys]) > 0:
            partial = True

        else:
            return_fn = True

    else:
        target = node.get(target_key)

    if node.get("__meta", None) is not None and node["__meta"].get(
        "import_pyfile", False
    ):
        obj = resolve_module_pyfile(
            node["__meta"]["qualname"], node["__meta"]["filepath"]
        )

    else:
        obj = resolve_module(target)

    signature = inspect.signature(obj)

    if key_key in node and node[key_key] in singleton_dict:
        return singleton_dict[node[key_key]]

    if args_key in node:
        args_node = node[args_key]

        if len(args_node) > len(args):
            args_init = []

            for a in args_node[len(args) :]:
                args_init.append(
                    instance_traverse(
                        a,
                        recursive=recursive,
                        instantiate=instantiate,
                        keyword_args=keyword_args,
                        _tags_=_tags_,
                        root=False,
                        singleton_dict=singleton_dict,
                    )
                )

            args = list(args) + args_init

    pos_replace = list(signature.parameters.keys())[: len(args)]

    kwargs = {}

    if root and keyword_args is not None:
        for k, v in keyword_args.items():
            kwargs[k] = v

    for k, v in node.items():
        if k in exclude_keys:
            continue

        if k in pos_replace:
            continue

        if root and keyword_args is not None and k in keyword_args:
            kwargs[k] = keyword_args[k]

            continue

        kwargs[k] = instance_traverse(
            v,
            recursive=recursive,
            instantiate=instantiate,
            keyword_args=keyword_args,
            _tags_=_tags_,
            root=False,
            singleton_dict=singleton_dict,
        )

    if return_fn:
        instance = obj

    elif partial:
        instance = functools.partial(obj, *args, **kwargs)

    else:
        instance = obj(*args, **kwargs)

    if key_key in node and node[key_key] not in SINGLETON:
        singleton_dict[node[key_key]] = instance

    return instance


def _instance_traverse_dict(
    node,
    *args,
    recursive=True,
    instantiate=False,
    keyword_args=None,
    _tags_=None,
    root=True,
    singleton_dict=None,
):
    target_key = "__target"
    init_key = "__init"
    fn_key = "__fn"
    tag_key = "__tag"

    if target_key in node or init_key in node or fn_key in node:
        if instantiate:
            return _instance_traverse_instantiate(
                node,
                *args,
                recursive=recursive,
                instantiate=instantiate,
                keyword_args=keyword_args,
                _tags_=_tags_,
                root=root,
                singleton_dict=singleton_dict,
            )

        else:
            return _instance_traverse_validate(node, recursive=recursive, _tags_=_tags_)

    elif tag_key in node:
        tag = node[tag_key]

        if _tags_ is not None and tag in _tags_:
            return _tags_[tag]

        else:
            raise ValueError(f"Tag '{tag}' not found in _tags_")

    else:
        mapping = {}

        for k, v in node.items():
            mapping[k] = instance_traverse(
                v,
                recursive=recursive,
                instantiate=instantiate,
                keyword_args=keyword_args,
                _tags_=_tags_,
                root=False,
                singleton_dict=singleton_dict,
            )

        return mapping


def instance_traverse(
    node,
    *args,
    recursive=True,
    instantiate=False,
    keyword_args=None,
    _tags_=None,
    root=True,
    singleton_dict=None,
):
    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        seq = [
            instance_traverse(
                i,
                recursive=recursive,
                instantiate=instantiate,
                keyword_args=keyword_args,
                _tags_=_tags_,
                root=False,
                singleton_dict=singleton_dict,
            )
            for i in node
        ]

        return seq

    if isinstance(node, collections.abc.Mapping):
        return _instance_traverse_dict(
            node,
            *args,
            recursive=recursive,
            instantiate=instantiate,
            keyword_args=keyword_args,
            _tags_=_tags_,
            root=root,
            singleton_dict=singleton_dict,
        )

    else:
        return node


def init_singleton(nodes):
    key_key = "__key"

    for node in nodes:
        if key_key not in node:
            continue

        node_key = node[key_key]

        if node_key in SINGLETON:
            continue

        restrict_node = {k: v for k, v in node.items() if k != key_key}
        instance_traverse(restrict_node)
        SINGLETON[node_key] = instance_traverse(restrict_node, instantiate=True)


class Instance(dict):
    @classmethod
    def __get_pydantic_core_schema__(self, cls, source_type):
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.dict_schema()
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v_new = instance_traverse(v)
        instance = cls(v_new)

        return instance

    def make(self, *args, _tags_=None, **kwargs):
        # init_singleton(flatten_tree(self))
        singleton_dict = {}

        return instance_traverse(
            self,
            *args,
            instantiate=True,
            keyword_args=kwargs,
            _tags_=_tags_,
            singleton_dict=singleton_dict,
        )

    def instantiate(self, *args, **kwargs):
        return self.make(*args, **kwargs)


def instantiate(instance, *args, _tags_: dict[str, Any] = None, **kwargs):
    """
    Instantiates an instance or traverses it if already instantiated.

    This function checks if the given instance has a 'make' method and attempts to instantiate it using that method.
    If the 'make' method is not present, it falls back to traversing the instance using the 'instance_traverse' function,
    ensuring that instantiation is attempted or the instance is properly traversed with the given arguments and keyword arguments.

    Parameters:
    - instance: The instance to be instantiated or traversed.
    - *args: Variable length argument list to be passed to the instantiation or traversal function.
    - _tags_: Optional tags to be considered during instantiation or traversal.
    - **kwargs: Arbitrary keyword arguments to be passed to the instantiation or traversal function.

    Returns:
    - The instantiated or traversed instance.
    """

    try:
        return instance.make(
            *args,
            _tags_=_tags_,
            **kwargs,
        )

    except AttributeError:
        # init_singleton(flatten_tree(instance))
        singleton_dict = {}

        return instance_traverse(
            instance,
            *args,
            instantiate=True,
            keyword_args=kwargs,
            _tags_=_tags_,
            singleton_dict=singleton_dict,
        )
