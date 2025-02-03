import collections
import functools
import inspect
from inspect import _ParameterKind
from copy import deepcopy
import sys
import typing
from typing import Any, Callable, Optional
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
    validator,
    ConfigDict,
    ValidatorFunctionWrapHandler,
    field_validator,
)

try:
    from pydantic_core import core_schema

except ImportError:
    pass

from slickconf.constants import (
    EXCLUDE_KEYS,
    TARGET_KEY,
    INIT_KEY,
    FN_KEY,
    VALIDATE_KEY,
    PARTIAL_KEY,
    REPEAT_KEY,
    ARGS_KEY,
    KEY_KEY,
    PLACEHOLDER_KEY,
    META_KEY,
    TAG_KEY,
    ANNOTATE_KEY,
)
from slickconf.container import NodeDict
from slickconf.pyconfig import resolve_module, resolve_module_pyfile

CONFIG_REGISTRY = {}
SINGLETON = {}


class Tag(dict):
    @classmethod
    def __get_pydantic_core_schema__(self, cls, source_type):
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.dict_schema()
        )

    @classmethod
    def validate(cls, v):
        if "__tag" not in v:
            raise ValueError(f"value with type of field or tag is required; got {v}")

        return cls(v)


class TaggableModel(BaseModel):
    def __init_subclass__(cls):
        super().__init_subclass__()
        annotations = getattr(cls, "__annotations__", {})
        for attr_name, attr_type in annotations.items():
            annotations[attr_name] = attr_type | Tag


class Config(BaseModel):
    class Config:
        extra = "forbid"
        protected_namespaces = ()


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


@classmethod
def instance_validator(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    try:
        return handler(value)

    except ValidationError as errors:
        filtered_errors = []

        for error in errors.errors():
            input = error["input"]
            is_instance = False
            if not isinstance(input, str):
                for exclude_key in EXCLUDE_KEYS:
                    if exclude_key in input:
                        is_instance = True

                        break

                if TAG_KEY in input:
                    is_instance = True

                    break

            if not is_instance:
                filtered_errors.append(error)

        if len(filtered_errors) > 0:
            raise ValidationError.from_exception_data(errors.title, filtered_errors)

        else:
            return value


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

        # pydantic do not want fields start with `_`
        # so we skip keys starts with `_`
        if k.startswith("_"):
            continue

        if (
            v.kind == v.VAR_POSITIONAL
            or v.kind == v.VAR_KEYWORD
            or v.kind == v.POSITIONAL_ONLY
        ):
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

    validators = {
        "params": _params,
        "make": _init_fn,
    }

    if len(params) > 0:
        validators["instance_validator"] = field_validator(*params.keys(), mode="wrap")(
            instance_validator
        )

    if type_name is not None:
        validators["check_type"] = _check_type(type_name)

    if strict:
        config = ConfigDict(
            extra="forbid", arbitrary_types_allowed=True, protected_namespaces=()
        )

    else:
        config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model = create_model(
        name,
        __config__=config,
        __validators__=validators,
        __module__=__name__,
        **params,
    )

    setattr(sys.modules[__name__], name, model)

    return model


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
    exclude_keys = {
        TARGET_KEY,
        INIT_KEY,
        FN_KEY,
        VALIDATE_KEY,
        PARTIAL_KEY,
        ARGS_KEY,
        KEY_KEY,
    }

    placeholders = set()

    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        seq = [find_placeholder(n) for i, n in enumerate(node)]

        for s in seq:
            placeholders = placeholders.union(s)

        return placeholders

    if isinstance(node, collections.abc.Mapping):
        if TARGET_KEY in node or INIT_KEY in node or FN_KEY in node:
            if ARGS_KEY in node:
                args_node = node[ARGS_KEY]

                for arg in args_node:
                    placeholders = placeholders.union(find_placeholder(arg))

            for k, v in node.items():
                if k in exclude_keys:
                    continue

                try:
                    if v[PLACEHOLDER_KEY] == PLACEHOLDER_KEY:
                        placeholders.add(k)

                        continue

                except:
                    pass

                placeholders = placeholders.union(find_placeholder(v))

    return placeholders


def bind_arguments_from_node(node, parameters):
    positional = []
    keyword = {}
    for key, val in node.items():
        if key in EXCLUDE_KEYS:
            continue

        keyword[key] = val

    arg_i = 0

    var_positional = None
    for i, param in enumerate(parameters.values()):
        if param.kind == _ParameterKind.VAR_POSITIONAL:
            var_positional = i

            break

    for i, param in enumerate(list(parameters.values())):
        if param.kind == _ParameterKind.POSITIONAL_ONLY:
            if arg_i < len(node[ARGS_KEY]):
                positional.append(node[ARGS_KEY][arg_i])
                arg_i += 1

        elif param.kind == _ParameterKind.POSITIONAL_OR_KEYWORD:
            if var_positional is not None and param.name in node:
                positional.append(node[param.name])
                del keyword[param.name]

    if var_positional is not None:
        for arg in node[ARGS_KEY][arg_i:]:
            positional.append(arg)

    if ARGS_KEY in keyword:
        del keyword[ARGS_KEY]

    return positional, keyword, var_positional


def _instance_traverse_validate(
    node,
    recursive=True,
    instantiate=False,
    _tags_=None,
):
    partial = node.get(PARTIAL_KEY, False)
    do_validate = node.get(VALIDATE_KEY, True)

    if INIT_KEY in node:
        target = node.get(INIT_KEY)

    elif FN_KEY in node:
        target = node.get(FN_KEY)

        if len([k for k in node if k not in EXCLUDE_KEYS]) > 0:
            partial = True

        else:
            do_validate = False

    else:
        target = node.get(TARGET_KEY)

    if node.get(META_KEY, None) is not None and node[META_KEY].get(
        "import_pyfile", False
    ):
        obj = resolve_module_pyfile(
            node[META_KEY]["qualname"], node[META_KEY]["filepath"]
        )

    else:
        obj = resolve_module(target)

    signature = inspect.signature(obj)

    rest = {}

    positional, keyword, _ = bind_arguments_from_node(node, signature.parameters)

    args_replaced = []
    if len(positional) > 0:
        for arg, (k, param) in zip(positional, signature.parameters.items()):
            if isinstance(arg, collections.abc.Mapping) and ANNOTATE_KEY in arg:
                arg = arg["value"]

            if (
                param.kind == _ParameterKind.POSITIONAL_ONLY
                or param.kind == _ParameterKind.VAR_POSITIONAL
            ):
                continue

            rest[k] = arg
            args_replaced.append(k)

    annotation_replaced = []

    for k, v in keyword.items():
        if isinstance(v, collections.abc.Mapping) and ANNOTATE_KEY in v:
            annotation_replaced.append(k)
            v = v["value"]

        rest[k] = instance_traverse(
            v, recursive=recursive, _tags_=_tags_, instantiate=instantiate
        )

        if k in args_replaced:
            raise TypeError(f"{target} got multiple values for argument '{k}'")

    if do_validate:
        name = "instance." + target

        exclude = []

        for r_k in rest.keys():
            # pydantic do not want fields start with `_`
            # so we skip keys starts with `_`
            if r_k.startswith("_"):
                exclude.append(r_k)

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

    for annotation in annotation_replaced:
        del rest[annotation]

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
    return_fn = False
    partial = node.get(PARTIAL_KEY, False)

    if INIT_KEY in node:
        target = node.get(INIT_KEY)

    elif FN_KEY in node:
        target = node.get(FN_KEY)

        if len([k for k in node if k not in EXCLUDE_KEYS]) > 0:
            partial = True

        else:
            return_fn = True

    else:
        target = node.get(TARGET_KEY)

    if node.get(META_KEY, None) is not None and node[META_KEY].get(
        "import_pyfile", False
    ):
        obj = resolve_module_pyfile(
            node["__meta"]["qualname"], node["__meta"]["filepath"]
        )

    else:
        obj = resolve_module(target)

    signature = inspect.signature(obj)

    if KEY_KEY in node and node[KEY_KEY] in singleton_dict:
        return singleton_dict[node[KEY_KEY]]

    positional, keyword, var_positional_id = bind_arguments_from_node(
        node, signature.parameters
    )

    if len(positional) > 0:
        if len(positional) > len(args):
            args_init = []

            for a in positional[len(args) :]:
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

    len_replaced = len(args)
    if var_positional_id is not None:
        len_replaced = min(len_replaced, var_positional_id + 1)

    pos_replace = list(signature.parameters.keys())[:len_replaced]

    kwargs = {}

    if root and keyword_args is not None:
        for k, v in keyword_args.items():
            kwargs[k] = v

    for k, v in keyword.items():
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

    if KEY_KEY in node and node[KEY_KEY] not in SINGLETON:
        singleton_dict[node[KEY_KEY]] = instance

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
    if TARGET_KEY in node or INIT_KEY in node or FN_KEY in node:
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

    elif REPEAT_KEY in node:
        node_dict = NodeDict._recursive_init_(node[REPEAT_KEY])
        nodes = [deepcopy(node_dict) for _ in range(node["times"])]

        return instance_traverse(
            nodes,
            recursive=recursive,
            instantiate=instantiate,
            keyword_args=keyword_args,
            _tags_=_tags_,
            root=False,
            singleton_dict=singleton_dict,
        )

    elif TAG_KEY in node:
        tag = node[TAG_KEY]

        if _tags_ is not None and tag in _tags_:
            return _tags_[tag]

        elif "default" in node:
            return instance_traverse(
                node["default"],
                recursive=recursive,
                instantiate=instantiate,
                keyword_args=keyword_args,
                _tags_=_tags_,
                root=False,
                singleton_dict=singleton_dict,
            )

        else:
            raise ValueError(f"Tag '{tag}' not found in _tags_")

    elif instantiate and ANNOTATE_KEY in node:
        return node["value"]

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


class Instance(NodeDict):
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

    if hasattr(instance, "make"):
        return instance.make(
            *args,
            _tags_=_tags_,
            **kwargs,
        )

    singleton_dict = {}

    return instance_traverse(
        instance,
        *args,
        instantiate=True,
        keyword_args=kwargs,
        _tags_=_tags_,
        singleton_dict=singleton_dict,
    )
