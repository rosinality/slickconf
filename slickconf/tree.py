import collections
from collections import deque, OrderedDict, defaultdict
import threading

NODE_REGISTRY_LOCK = threading.Lock()
SUPPORTED_NODES = {}
STANDARD_DICT_TYPES = frozenset({dict, OrderedDict, defaultdict})
BUILTIN_TYPES = frozenset({tuple, list, dict})


def register_pytree_node(cls, flatten_fn, unflatten_fn, *, flatten_with_keys_fn=None):
    with NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            raise ValueError(f"Node {cls} already registered")

    _register_pytree_node(
        cls, flatten_fn, unflatten_fn, flatten_with_keys_fn=flatten_with_keys_fn
    )


class NodeDef:
    def __init__(self, cls, flatten_fn, unflatten_fn, flatten_with_keys_fn=None):
        self.cls = cls
        self.flatten_fn = flatten_fn
        self.unflatten_fn = unflatten_fn
        self.flatten_with_keys_fn = flatten_with_keys_fn


def _register_pytree_node(cls, flatten_fn, unflatten_fn, *, flatten_with_keys_fn=None):
    with NODE_REGISTRY_LOCK:
        node_def = NodeDef(cls, flatten_fn, unflatten_fn, flatten_with_keys_fn)
        SUPPORTED_NODES[cls] = node_def


class SequenceKey:
    def __init__(self, idx):
        self.idx = idx

    def __repr__(self):
        return f"[{self.idx!r}]"

    def get(self, sequence):
        return sequence[self.idx]


class MappingKey:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"[{self.key!r}]"

    def get(self, mapping):
        return mapping[self.key]


class GetAttrKey:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f".{self.name!r}"

    def get(self, obj):
        return getattr(obj, self.name)


def _tuple_flatten(d):
    return list(d), None


def _tuple_flatten_with_keys(d):
    values, context = _tuple_flatten(d)

    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _tuple_unflatten(values, context):
    return tuple(values)


def _list_flatten(d):
    return d, None


def _list_flatten_with_keys(d):
    values, context = _list_flatten(d)

    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _list_unflatten(values, context):
    return list(values)


def _dict_flatten(d):
    return list(d.values()), list(d.keys())


def _dict_flatten_with_keys(d):
    values, context = _dict_flatten(d)

    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _dict_unflatten(values, context):
    return dict(zip(context, values))


_register_pytree_node(
    tuple,
    _tuple_flatten,
    _tuple_unflatten,
    flatten_with_keys_fn=_tuple_flatten_with_keys,
)
_register_pytree_node(
    list, _list_flatten, _list_unflatten, flatten_with_keys_fn=_list_flatten_with_keys
)
_register_pytree_node(
    dict, _dict_flatten, _dict_unflatten, flatten_with_keys_fn=_dict_flatten_with_keys
)


def _get_node_type(tree):
    return type(tree)


def _is_leaf(tree, is_leaf=None):
    return (is_leaf is not None and is_leaf(tree)) or _get_node_type(
        tree
    ) not in SUPPORTED_NODES


class TreeSpec:
    def __init__(self, type, context, children_specs):
        self.type = type
        self.context = context
        self.children_specs = children_specs

        self.n_nodes = sum((spec.n_nodes for spec in children_specs), start=1)
        self.n_leaves = sum(spec.n_leaves for spec in children_specs)
        self.n_children = len(children_specs)

    def __repr__(self, indent: int = 0):
        repr_prefix = f"TreeSpec({self.type.__name__}, {self.context}, ["
        children_specs_str = ""

        if self.n_children > 0:
            indent += 2
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += "," if self.n_children > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self.children_specs[1:]
                ]
            )

        repr_suffix = f"{children_specs_str}])"

        return repr_prefix + repr_suffix

    def is_leaf(self):
        return self.n_nodes == 1 and self.n_leaves == 1

    def flatten_up_to(self, tree):
        def helper(treespec, tree, subtrees):
            if treespec.is_leaf():
                subtrees.append(tree)

                return

            node_type = _get_node_type(tree)

            if treespec.type not in BUILTIN_TYPES:
                if node_type != treespec.type:
                    raise ValueError(
                        f"Incompatible types: {treespec.type} and {node_type}"
                    )

                flatten_fn = SUPPORTED_NODES[treespec.type].flatten_fn
                children, context = flatten_fn(tree)

                if len(children) != treespec.n_children:
                    raise ValueError(
                        f"Expected {treespec.n_children} children, but got {len(children)}"
                    )

                if context != treespec.context:
                    raise ValueError(
                        f"Incompatible contexts for node type {treespec.type!r}; {context} and {treespec.context}"
                    )

            else:
                both_standard_dict = (
                    treespec.type in STANDARD_DICT_TYPES
                    and node_type in STANDARD_DICT_TYPES
                )

                if not both_standard_dict and node_type != treespec.type:
                    raise ValueError(
                        f"Incompatible types: {treespec.type} and {node_type}"
                    )

                if len(tree) != treespec.n_children:
                    raise ValueError(
                        f"Expected {treespec.n_children} children, but got {len(tree)}"
                    )

                if both_standard_dict:
                    dict_context = (
                        treespec.context
                        if treespec.type is not defaultdict
                        else treespec.context[1]
                    )
                    expected_keys = dict_context
                    got_key_set = set(tree)
                    expected_key_set = set(expected_keys)

                    if got_key_set != expected_key_set:
                        missing_keys = expected_key_set.difference(got_key_set)
                        extra_keys = got_key_set.difference(expected_key_set)
                        message = ""

                        if missing_keys:
                            message += f"; missing keys: {missing_keys}"

                        if extra_keys:
                            message += f"; extra keys: {extra_keys}"

                        raise ValueError(f"Incompatible dictionaries: {message}")

                    children = [tree[key] for key in expected_keys]

                else:
                    flatten_fn = SUPPORTED_NODES[treespec.type].flatten_fn
                    children, context = flatten_fn(tree)

                    if len(children) != treespec.n_children:
                        raise ValueError(
                            f"Expected {treespec.n_children} children, but got {len(children)}"
                        )

                    if node_type is not deque and context != treespec.context:
                        raise ValueError(
                            f"Incompatible contexts for node type {treespec.type!r}; {context} and {treespec.context}"
                        )

            for subtree, subspec in zip(children, treespec.children_specs):
                helper(subspec, subtree, subtrees)

        subtrees = []
        helper(self, tree, subtrees)

        return subtrees

    def unflatten(self, leaves):
        if not isinstance(leaves, (list, tuple)):
            leaves = list(leaves)

        if len(leaves) != self.n_leaves:
            raise ValueError(f"Expected {self.n_leaves} leaves, but got {len(leaves)}")

        if self.is_leaf():
            return leaves[0]

        unflatten_fn = SUPPORTED_NODES[self.type].unflatten_fn

        start = 0
        end = 0
        child_pytrees = []

        for child_spec in self.children_specs:
            end += child_spec.n_leaves
            child_pytrees.append(child_spec.unflatten(leaves[start:end]))
            start = end

        return unflatten_fn(child_pytrees, self.context)


class LeafSpec(TreeSpec):
    def __init__(self):
        super().__init__(None, None, [])

        self.n_nodes = 1
        self.n_leaves = 1
        self.n_children = 0

    def __repr__(self, indent=0):
        return "*"


LEAF_SPEC = LeafSpec()


def tree_flatten(tree, is_leaf=None):
    def helper(node, leaves):
        if _is_leaf(node, is_leaf):
            leaves.append(node)

            return LEAF_SPEC

        node_type = _get_node_type(node)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        children, context = flatten_fn(node)

        subspecs = [helper(child, leaves) for child in children]

        return TreeSpec(node_type, context, subspecs)

    leaves = []
    treespec = helper(tree, leaves)

    return leaves, treespec


def _generate_key_paths(key_path, tree, is_leaf=None):
    if is_leaf and is_leaf(tree):
        yield key_path, tree

        return

    node_type = _get_node_type(tree)
    handler = SUPPORTED_NODES.get(node_type)

    if not handler:
        yield key_path, tree

        return

    flatten_with_keys = handler.flatten_with_keys_fn

    if flatten_with_keys:
        key_children, _ = flatten_with_keys(tree)

        for k, c in key_children:
            yield from _generate_key_paths((*key_path, k), c, is_leaf)

    else:
        raise ValueError(f"No flatten_with_keys_fn for node type {node_type!r}")


def tree_flatten_with_path(tree, is_leaf=None):
    _, treespec = tree_flatten(tree, is_leaf=is_leaf)

    return list(_generate_key_paths((), tree, is_leaf)), treespec


def tree_flatten_exactly_one_level(tree):
    paths_and_subtrees, treespec = tree_flatten_with_path(
        tree, is_leaf=lambda subtree: subtree is not tree
    )

    if treespec == LEAF_SPEC:
        return None

    keys_and_subtrees = [(key, subtree) for ((key,), subtree) in paths_and_subtrees]

    return keys_and_subtrees, treespec


def tree_unflatten(leaves, treespec):
    if not isinstance(treespec, TreeSpec):
        raise TypeError("treespec must be a TreeSpec instance")

    return treespec.unflatten(leaves)


def tree_map(fn, tree, *rests, is_leaf=None):
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(rest) for rest in rests]

    return treespec.unflatten(map(fn, *flat_args))


def tree_map_with_path(fn, tree, *rests, is_leaf=None):
    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf=is_leaf)
    keypath_leaves = list(zip(*keypath_leaves))
    all_keypath_leaves = keypath_leaves + [
        treespec.flatten_up_to(rest) for rest in rests
    ]

    return treespec.unflatten(fn(*xs) for xs in zip(*all_keypath_leaves))


def traverse(node):
    if isinstance(node, collections.abc.Sequence) and not isinstance(node, str):
        for n in node:
            yield from traverse(n)

    elif isinstance(node, collections.abc.Mapping):
        for v in node.values():
            yield from traverse(v)

        yield node
