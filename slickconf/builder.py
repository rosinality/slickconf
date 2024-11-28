import ast
import builtins
import copy
import functools
import importlib
import os
import inspect
import pydoc
import textwrap
import types
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Tuple, Union

from slickconf.config import resolve_module

CFG_PACKAGE_NAME = "slickconf._conf_loader"


def str_to_import(name: str):
    obj = pydoc.locate(name)

    if obj is None:
        obj = resolve_module(name)

    return obj


def validate_syntax(filename: str):
    with open(filename) as f:
        code = f.read()

    try:
        ast.parse(code)

    except SyntaxError as e:
        raise SyntaxError(f"{filename} has syntax error") from e


def random_package_name(filename: str):
    # generate a random package name when loading config files
    return CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


def import_to_str(obj: object):
    module, qualname = obj.__module__, obj.__qualname__

    module_parts = module.split(".")

    for i in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:i])
        candid = f"{prefix}.{qualname}"

        try:
            if str_to_import(candid) is obj:
                return candid

        except ImportError:
            pass

    return f"{module}.{qualname}"


CALL_HANDLER_ID = "__auto_config_call_handler__"
CLOSURE_WRAPPER_ID = "__auto_config_closure_wrapper__"
EMPTY_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
)


def auto_config_call_handler(fn_or_cls: callable, *args, **kwargs):
    if (
        fn_or_cls is field
        or fn_or_cls is finalize
        or fn_or_cls is tag
        or fn_or_cls is single
        or fn_or_cls is function
        or isinstance(fn_or_cls, EagerCallContainer)
        or fn_or_cls is copy.copy
        or fn_or_cls is copy.deepcopy
        or inspect.isbuiltin(fn_or_cls)
        or (inspect.isclass(fn_or_cls) and fn_or_cls.__module__ == "builtins")
    ):
        return fn_or_cls(*args, **kwargs)

    if fn_or_cls is functools.partial:
        return F[args[0]](*args[1:], **kwargs)

    return single[fn_or_cls](*args, **kwargs)


class AutoConfigNodeTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        return ast.Call(
            func=ast.Name(id=CALL_HANDLER_ID, ctx=ast.Load()),
            args=[node.func, *(self.visit(arg) for arg in node.args)],
            keywords=[self.visit(keyword) for keyword in node.keywords],
        )


def _wrap_ast_for_fn_with_closure_vars(
    module: ast.Module,
    fn: types.FunctionType,
) -> ast.Module:
    """Wraps `module.body` in a function that defines closure variables for `fn`.

    If `fn` has any free variables (i.e., it's `__code__.co_freevars` is not
    empty), we want to make sure that compiling its AST (assumed to be in the body
    of `module`) will create the same set of free variables in the resulting code
    object. However, by default this won't happen, since we would be compiling
    `fn`'s AST in the absence of its original context (e.g., just compiling a
    nested function, and not the containing one).

    To work around this issue, this function wraps `module.body` in another
    `FunctionDef` that defines dummy variables corresponding to `fn`'s free
    variables. This causes the subsequent compile step to create the right set of
    free variables, and allows us to use `fn.__closure__` when creating a
    new function object via `types.FunctionType`.

    We also add <_CALL_HANDLER_ID> as a final dummy variable, and append its value
    (the call handler) to `fn.__closure__` when creating the new function object.

    Effectively, this wrapping looks like the following Python code:

        def __auto_config_closure_wrapper__():
          closure_var_1 = None
          closure_var_2 = None
          ...
          <_CALL_HANDLER_ID> = None

          def fn(...):  # Or some expression involving a lambda.
            ...  # Contains references to the closure variables.

    Args:
      module: An `ast.Module` object whose body contains the function definition
        for `fn` (e.g., as an `ast.FunctionDef` or `ast.Lambda`).
      fn: The function to create dummy closure variables for (assumed to
        correspond to the body of `module`).

    Returns:
      A new `ast.Module` containing an additional wrapper `ast.FunctionDef` that
      defines dummy closure variables.
    """
    ast_name = lambda name: ast.Name(id=name, ctx=ast.Store())
    ast_none = ast.Constant(value=None)
    closure_var_definitions = [
        ast.Assign(targets=[ast_name(var_name)], value=ast_none)
        for var_name in fn.__code__.co_freevars + (CALL_HANDLER_ID,)
    ]

    wrapper_module = ast.Module(
        body=[
            ast.FunctionDef(
                name=CLOSURE_WRAPPER_ID,
                args=EMPTY_ARGUMENTS,
                body=[
                    *closure_var_definitions,
                    *module.body,
                ],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    wrapper_module = ast.fix_missing_locations(wrapper_module)
    return wrapper_module


def _find_function_code(code: types.CodeType, fn_name: str):
    """Finds the code object within `code` corresponding to `fn_name`."""
    code = [
        const
        for const in code.co_consts
        if inspect.iscode(const) and const.co_name == fn_name
    ]
    assert len(code) == 1, f"Couldn't find function code for {fn_name!r}."
    return code[0]


def _unwrap_code_for_fn(code: types.CodeType, fn: types.FunctionType):
    """Unwraps `code` to find the code object for `fn`.

    This function assumes `code` is the result of compiling an `ast.Module`
    returned by `_wrap_node_for_fn_with_closure_vars`.

    Args:
      code: A code object containing code for `fn`.
      fn: The function to find a code object for within `code`.

    Returns:
      The code object corresponding to `fn`.
    """
    code = _find_function_code(code, CLOSURE_WRAPPER_ID)
    code = _find_function_code(code, fn.__name__)
    return code


def _make_closure_cell(contents):
    """Returns `types.CellType(contents)`."""
    if hasattr(types, "CellType"):
        # `types.CellType` added in Python 3.8.
        return types.CellType(contents)  # pytype: disable=wrong-arg-count
    else:
        # For earlier versions of Python, build a dummy function to get CellType.
        dummy_fn = lambda: contents
        cell_type = type(dummy_fn.__closure__[0])
        return cell_type(contents)


def config_fn(fn):
    filename = inspect.getsourcefile(fn)
    line_number = fn.__code__.co_firstlineno
    source = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(source)
    node = AutoConfigNodeTransformer().visit(node)
    node = ast.fix_missing_locations(node)
    node = ast.increment_lineno(node, line_number - 1)
    node = _wrap_ast_for_fn_with_closure_vars(node, fn)
    code = compile(node, filename, "exec")
    code = _unwrap_code_for_fn(code, fn)

    indexed_handlers = []
    closure = list(fn.__closure__ or ())
    if CALL_HANDLER_ID in code.co_freevars:
        handler_idx = code.co_freevars.index(CALL_HANDLER_ID)
        handler = _make_closure_cell(auto_config_call_handler)
        indexed_handlers.append((handler_idx, handler))

    for handler_idx, handler in sorted(indexed_handlers):
        closure.insert(handler_idx, handler)
    closure = tuple(closure)

    auto_config_fn = types.FunctionType(code, fn.__globals__, closure=closure)
    auto_config_fn.__defaults__ = fn.__defaults__
    auto_config_fn.__kwdefaults__ = fn.__kwdefaults__

    return auto_config_fn


def auto_config_from_source(source: str, filename: str):
    source = textwrap.dedent(source)
    node = ast.parse(source)
    node = AutoConfigNodeTransformer().visit(node)
    node = ast.fix_missing_locations(node)
    code = compile(node, filename, "exec")

    return code


def finalize(node, *args, **kwargs):
    node = deepcopy(node)

    if "__fn" in node:
        name = node["__fn"]
        del node["__fn"]
        node["__init"] = name

    if len(args) > 0:
        if "__args" in node:
            node["__args"] += list(args)

        else:
            node["__args"] = list(args)

    node = {**node, **kwargs}

    return node


class NodeDict(dict):
    def __copy__(self):
        new_dict = NodeDict()
        for k, v in self.items():
            if k == "__key":
                obj, _ = v.rsplit("#", 1)
                new_dict[k] = obj + f"#{single.counter}"
                single.counter += 1

            else:
                new_dict[k] = copy.deepcopy(v)

        return NodeDict(new_dict)

    def __deepcopy__(self, memo):
        new_dict = NodeDict()
        memo[id(self)] = new_dict

        for k, v in self.items():
            if k == "__key":
                obj, _ = v.rsplit("#", 1)
                v = obj + f"#{Single.counter}"
                Single.counter += 1

            new_dict[copy.deepcopy(k, memo)] = copy.deepcopy(v, memo)

        return NodeDict(new_dict)


def build(__key, __name, *args, **kwargs):
    node = {__key: __name}

    if len(args) > 0:
        node["__args"] = list(args)

    node = NodeDict({**node, **kwargs})

    return node


def build_init(__name, *args, **kwargs):
    return build("__init", __name, *args, **kwargs)


def build_fn(__name, *args, **kwargs):
    return build("__fn", __name, *args, **kwargs)


def placeholder(default=None):
    return build("__placeholder", "__placeholder", default=default)


def tag(tag_name, default=None):
    return build("__tag", tag_name, default=default)


def function(obj, *args, **kwargs):
    filepath = None
    qualname = None

    if not isinstance(obj, str):
        filepath = obj.__code__.co_filename
        qualname = obj.__qualname__
        obj = import_to_str(obj)

    res = build_fn(obj, *args, **kwargs)

    if filepath is not None:
        res["__meta"] = {
            "filepath": filepath,
            "qualname": qualname,
            "import_pyfile": True,
        }

    return res


class Init:
    def __init__(self, name, fn=False, key=None, filepath=None):
        self.name = name
        self.fn = fn
        self.key = key
        self.filepath = filepath

    def __call__(self, *args, **kwargs):
        if self.fn:
            res = build_fn(self.name, *args, **kwargs)

            if self.filepath is not None:
                res["_meta_"] = {"filepath": self.filepath}

            return res
        
        res = build_init(self.name, *args, **kwargs)

        if self.key is not None:
            res["__key"] = self.key

        if self.filepath is not None:
            res["_meta_"] = {"filepath": self.filepath}

        return res


class Single:
    counter = 0

    def __getitem__(self, obj):
        fn = False

        if not isinstance(obj, str):
            obj = import_to_str(obj)

        key = f"{obj}#{Single.counter}"
        Single.counter += 1

        return Init(obj, fn, key=key)


class EagerCallContainer:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class EagerCall:
    def __getitem__(self, obj):
        return EagerCallContainer(obj)


class LazyCall:
    def __getitem__(self, obj):
        fn = False
        
        if isinstance(obj, tuple):
            obj, fn = obj

        if not isinstance(obj, str):
            obj = import_to_str(obj)

        return Init(obj, fn)


class LazyFn:
    def __getitem__(self, obj):
        if not isinstance(obj, str):
            obj = import_to_str(obj)

        return Init(obj, True)


@contextmanager
def patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
        e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)

        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)

        cur_name = relative_import_path.lstrip(".")

        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)

        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"

        if not os.path.isfile(cur_file):
            raise ImportError(
                f"cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist"
            )

        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            validate_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(
                random_package_name(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file

            with open(cur_file) as f:
                content = f.read()

            exec(compile(content, cur_file, "exec"), module.__dict__)

            # for name in fromlist:  # turn imported dict into DictConfig automatically
            #     val = _cast_to_config(module.__dict__[name])
            #     module.__dict__[name] = val

            return module

        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class PyConfig:
    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.
        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant
        filename = os.path.abspath(filename)

        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")

        if filename.endswith(".py"):
            validate_syntax(filename)

            with patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": random_package_name(filename),
                }
                with open(filename) as f:
                    content = f.read()

                if content.strip().startswith('"use legacy"'):
                    # Compile first with filename to:
                    # 1. make filename appears in stacktrace
                    # 2. make load_rel able to find its parent's (possibly remote) location
                    exec(compile(content, filename, "exec"), module_namespace)

                else:
                    module_namespace[CALL_HANDLER_ID] = auto_config_call_handler

                    exec(auto_config_from_source(content, filename), module_namespace)

            ret = module_namespace

        ret = ret["conf"].to_dict()

        if has_keys:
            return tuple(ret[a] for a in keys)

        return ret


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
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)

        except AttributeError:
            try:
                return self[key]

            except KeyError:
                raise AttributeError(key)

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)

        except AttributeError:
            try:
                self[key] = value

            except:
                raise AttributeError(key)

        else:
            object.__setattr__(self, key, value)

    def __delattr__(self, key):
        try:
            object.__getattribute__(self, key)

        except AttributeError:
            try:
                del self[key]

            except KeyError:
                raise AttributeError(key)

        else:
            object.__delattr__(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    def to_dict(self):
        return unfold_field(self)


L = LazyCall()
F = LazyFn()
field = Field
single = Single()
call = EagerCall()
