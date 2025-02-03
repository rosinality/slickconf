import builtins
import contextlib
import functools
import importlib

from slickconf.container import NodeDictProxyObject


@contextlib.contextmanager
def imports():
    new_import = functools.partial(
        _fake_import, proxy_cls=NodeDictProxyObject, origin_import=builtins.__import__
    )
    lazy_cm = contextlib.nullcontext()

    with lazy_cm, _fake_imports(new_import=new_import):
        try:
            yield

        except ImportError as e:
            raise e


@contextlib.contextmanager
def _fake_imports(new_import):
    origin_import = builtins.__import__

    try:
        builtins.__import__ = new_import
        yield

    finally:
        builtins.__import__ = origin_import


def _maybe_import(origin_import, module_name, from_name=None):
    with _fake_imports(new_import=origin_import):
        fromlist = ()

        if from_name is not None:
            fromlist = (from_name,)

        return origin_import(module_name, fromlist=fromlist)


def _fake_import(
    name, globals_, locals_, fromlist=(), level=0, *, proxy_cls, origin_import
):
    del globals_, locals_

    root_name, *parts = name.split(".")
    root = proxy_cls.from_cache(name=root_name)
    root.is_import = True

    if not fromlist:
        child = root

        for name in parts:
            child = child.child_import(name)

        child.set_target(_maybe_import(origin_import, child.qualname))

    else:
        for name in parts:
            root = root.child_import(name)

        for name in fromlist:
            child = root.child_import(name)
            target = _maybe_import(origin_import, root.qualname, name)
            target = getattr(target, name)
            child.set_target(target)

    return root
