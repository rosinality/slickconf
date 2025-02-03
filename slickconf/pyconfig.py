import os
import pydoc


def str_to_import(name: str):
    obj = pydoc.locate(name)

    if obj is None:
        obj = resolve_module(name)

    return obj


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
