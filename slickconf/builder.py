import ast
import builtins
import copy
import functools
import importlib
import os
import inspect
import linecache
import textwrap
import types
import uuid
from contextlib import contextmanager
from copy import deepcopy
from typing import Tuple, Union

import libcst as cst

from slickconf.constants import (
    INIT_KEY,
    FN_KEY,
    ARGS_KEY,
    REPEAT_KEY,
)
from slickconf.container import Field, NodeDict, SingleCounter
from slickconf.pyconfig import import_to_str

CFG_PACKAGE_NAME = "slickconf._conf_loader"


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


CALL_HANDLER_ID = "__auto_config_call_handler__"
CLOSURE_WRAPPER_ID = "__auto_config_closure_wrapper__"
EXEMPT_DECORATOR_ID = "__auto_config_exempt_decorator__"
EMPTY_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
)


def exempt(fn_or_cls: callable):
    fn_or_cls.__exempt__ = True

    return fn_or_cls


def get_instance_attr(instance, attr):
    return getattr(instance, attr)


def auto_config_call_handler(fn_or_cls: callable, *args, **kwargs):
    if (
        fn_or_cls
        in {
            Field,
            finalize,
            tag,
            single,
            function,
            annotate,
            copy.copy,
            copy.deepcopy,
            repeat,
        }
        or isinstance(fn_or_cls, EagerCallContainer)
        or inspect.isbuiltin(fn_or_cls)
        or (inspect.isclass(fn_or_cls) and fn_or_cls.__module__ == "builtins")
        or getattr(fn_or_cls, "__exempt__", False)
    ):
        return fn_or_cls(*args, **kwargs)

    if fn_or_cls is functools.partial:
        return F[args[0]](*args[1:], **kwargs)

    return single[fn_or_cls](*args, **kwargs)


def auto_config_exempt_decorator(fn):
    fn.__exempt__ = True

    return fn


class AutoConfigNodeTransformer(ast.NodeTransformer):
    def __init__(self, apply_exempt_decorator: bool = False):
        self.apply_exempt_decorator = apply_exempt_decorator

    def visit_Call(self, node):
        return ast.Call(
            func=ast.Name(id=CALL_HANDLER_ID, ctx=ast.Load()),
            args=[node.func, *(self.visit(arg) for arg in node.args)],
            keywords=[self.visit(keyword) for keyword in node.keywords],
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.apply_exempt_decorator:
            node.decorator_list.append(ast.Name(id=EXEMPT_DECORATOR_ID, ctx=ast.Load()))

        node = self.generic_visit(node)

        return node


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


def _is_lambda(fn):
    if not inspect.isfunction(fn):
        return False

    if not (hasattr(fn, "__name__") and hasattr(fn, "__code__")):
        return False

    return (fn.__name__ == "<lambda>") or (fn.__code__.co_name == "<lambda>")


class _LambdaFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, lambda_fn):
        super().__init__()

        self.lambda_fn = lambda_fn
        self.lineno = lambda_fn.__code__.co_firstlineno
        self.candidates = []

    def visit_Lambda(self, node):
        loc = self.get_metadata(cst.metadata.PositionProvider, node)

        if loc.start.line == self.lineno:
            self.candidates.append(node)


def _getsource_for_lambda(fn):
    module = inspect.getmodule(fn)
    filename = inspect.getsourcefile(fn)
    lines = linecache.getlines(filename, module.__dict__)
    source = "".join(lines)

    module_cst = cst.parse_module(source)
    lambda_finder = _LambdaFinder(fn)
    cst.metadata.MetadataWrapper(module_cst).visit(lambda_finder)

    if len(lambda_finder.candidates) == 1:
        lambda_node = lambda_finder.candidates[0]

        return cst.Module(body=[lambda_node]).code

    elif not lambda_finder.candidates:
        raise ValueError(f"Cannot find source for {fn} on line {lambda_finder.lineno}")

    else:
        raise ValueError(
            "Cannot find source for {fn} on line {lambda_finder.lineno}; multiple lambdas found"
        )


def config_fn(fn):
    filename = inspect.getsourcefile(fn)
    line_number = fn.__code__.co_firstlineno

    if _is_lambda(fn):
        source = _getsource_for_lambda(fn)

    else:
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
    auto_config_fn.__exempt__ = True

    return auto_config_fn


def auto_config_from_source(source: str, filename: str):
    source = textwrap.dedent(source)
    node = ast.parse(source)
    node = AutoConfigNodeTransformer(apply_exempt_decorator=True).visit(node)
    node = ast.fix_missing_locations(node)
    code = compile(node, filename, "exec")

    return code


def finalize(node, *args, **kwargs):
    node = deepcopy(node)

    if "__fn" in node:
        name = node[FN_KEY]
        del node[FN_KEY]
        node[INIT_KEY] = name

    if len(args) > 0:
        if ARGS_KEY in node:
            node[ARGS_KEY] += list(args)

        else:
            node[ARGS_KEY] = list(args)

    node = {**node, **kwargs}

    return node


def build_init(__name, __obj, *args, **kwargs):
    return NodeDict.build(INIT_KEY, __name, __obj, args, kwargs)


def build_fn(__name, __obj, *args, **kwargs):
    return NodeDict.build(FN_KEY, __name, __obj, args, kwargs)


def placeholder(default=None):
    return Field({"__placeholder": "__placeholder", "default": default})


def annotate(annotation, value):
    return Field({"__annotate": annotation, "value": value})


def repeat(node, times):
    return Field({REPEAT_KEY: node, "times": times})


class NoValue:
    def __repr__(self):
        return "slickconf.NO_VALUE"


NO_VALUE = NoValue()


def tag(tag_name, default=NO_VALUE):
    if isinstance(default, NoValue):
        return Field({"__tag": tag_name})

    return Field({"__tag": tag_name, "default": default})


def function(obj, *args, **kwargs):
    filepath = None
    qualname = None

    name = obj
    if not isinstance(obj, str):
        filepath = obj.__code__.co_filename
        qualname = obj.__qualname__
        name = import_to_str(obj)

    if CLOSURE_WRAPPER_ID in name:
        prefix = f"{CLOSURE_WRAPPER_ID}.<locals>."
        name = name.replace(prefix, "")
        qualname = qualname.replace(prefix, "")

    res = build_fn(name, obj, *args, **kwargs)

    if filepath is not None:
        res["__meta"] = {
            "filepath": filepath,
            "qualname": qualname,
            "import_pyfile": True,
        }

    return res


class Init:
    def __init__(self, obj, name, fn=False, key=None, filepath=None):
        self.obj = obj
        self.name = name
        self.fn = fn
        self.key = key
        self.filepath = filepath

    def __call__(self, *args, **kwargs):
        if self.fn:
            res = build_fn(self.name, self.obj, *args, **kwargs)

            if self.filepath is not None:
                res["_meta_"] = {"filepath": self.filepath}

            return res

        res = build_init(self.name, self.obj, *args, **kwargs)

        if self.key is not None:
            res["__key"] = self.key

        if self.filepath is not None:
            res["_meta_"] = {"filepath": self.filepath}

        return res


class Single:
    def __getitem__(self, obj):
        fn = False

        name = obj
        if not isinstance(obj, str):
            name = import_to_str(obj)

        key = f"{name}#{SingleCounter.counter}"
        SingleCounter.increase()

        return Init(obj, name, fn, key=key)


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

        name = obj
        if not isinstance(obj, str):
            name = import_to_str(obj)

        return Init(obj, name, fn)


class LazyFn:
    def __getitem__(self, obj):
        name = obj
        if not isinstance(obj, str):
            name = import_to_str(obj)

        return Init(obj, name, True)


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

            # exec(compile(content, cur_file, "exec"), module.__dict__)
            module.__dict__[CALL_HANDLER_ID] = auto_config_call_handler
            module.__dict__[EXEMPT_DECORATOR_ID] = auto_config_exempt_decorator
            exec(auto_config_from_source(content, cur_file), module.__dict__)

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
    def load(
        filename: str,
        keys: Union[None, str, Tuple[str, ...]] = None,
        config_name="conf",
    ):
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
                    module_namespace[EXEMPT_DECORATOR_ID] = auto_config_exempt_decorator

                    exec(auto_config_from_source(content, filename), module_namespace)

            ret = module_namespace

        ret = ret[config_name]

        if callable(ret):
            ret = ret()

        ret = ret.to_dict()

        if has_keys:
            return tuple(ret[a] for a in keys)

        return ret


L = LazyCall()
F = LazyFn()
single = Single()
call = EagerCall()
