import inspect
import linecache

import libcst as cst


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


def getsource_for_lambda(fn):
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
