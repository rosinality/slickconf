import ast
import math
import operator

BIN_OPERATORS = {
    # Arithmetic
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,  # ** operator
    # Bitwise
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

COMP_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda x, y: x in y,
    ast.NotIn: lambda x, y: x not in y,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,  # ~x
}

BOOL_OPERATORS = {
    ast.And: lambda values: all(values),
    ast.Or: lambda values: any(values),
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

# Safe constants
SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "True": True,
    "False": False,
    "None": None,
}


def evaluate(expression):
    """Evaluate a mathematical expression."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval(tree.body)

    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")


def _eval(node):
    """Recursively evaluate AST nodes."""

    # Numbers
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value

    # Names (variables/constants)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        else:
            raise ValueError(f"Name '{node.id}' is not allowed")

    # Binary operations
    elif isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        op_func = BIN_OPERATORS.get(type(node.op))
        if op_func:
            return op_func(left, right)
        else:
            raise ValueError(f"Binary operator {type(node.op).__name__} not allowed")

    elif isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        op_func = UNARY_OPERATORS.get(type(node.op))
        if op_func:
            return op_func(operand)
        else:
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")

    elif isinstance(node, ast.Compare):
        left = _eval(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator)
            op_func = COMP_OPERATORS.get(type(op))
            if op_func:
                result = op_func(left, right)
                if not result:
                    return False
                left = right
            else:
                raise ValueError(f"Comparison operator {type(op).__name__} not allowed")
        return True

    elif isinstance(node, ast.BoolOp):
        values = [_eval(value) for value in node.values]
        op_func = BOOL_OPERATORS.get(type(node.op))
        if op_func:
            return op_func(values)
        else:
            raise ValueError(f"Boolean operator {type(node.op).__name__} not allowed")

    # If expressions (ternary operator)
    elif isinstance(node, ast.IfExp):
        test = _eval(node.test)
        if test:
            return _eval(node.body)
        else:
            return _eval(node.orelse)

    elif isinstance(node, ast.List):
        return [_eval(elem) for elem in node.elts]

    elif isinstance(node, ast.Tuple):
        return tuple(_eval(elem) for elem in node.elts)

    elif isinstance(node, ast.Dict):
        return {_eval(key): _eval(value) for key, value in zip(node.keys, node.values)}

    elif isinstance(node, ast.Set):
        return {_eval(elem) for elem in node.elts}

    # Function calls
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in SAFE_FUNCTIONS:
                args = [_eval(arg) for arg in node.args]
                return SAFE_FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"Function '{func_name}' is not allowed")
        else:
            raise ValueError("Complex function calls are not allowed")

    # Attribute access (limited)
    elif isinstance(node, ast.Attribute):
        # Could add safe attribute access here if needed
        raise ValueError("Attribute access is not allowed")

    else:
        raise ValueError(f"Unsupported operation: {type(node).__name__}")
