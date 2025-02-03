import collections
import contextlib
from typing import Any, Collection

from slickconf.builder import config_fn, import_to_str, exempt, call
from slickconf.constants import INIT_KEY, FN_KEY, TARGET_KEY, TAG_KEY
from slickconf import tree as tree_fn


class SelectionHole:
    def __init__(self, path):
        self.path = path


class SelectionQuote:
    def __init__(self, quoted):
        self.quoted = quoted


class _SelectionBoundary:
    def __init__(self, selected):
        self.selected = selected


class _LeafWrapper:
    def __init__(self, wrapped_leaf):
        self.wrapped_leaf = wrapped_leaf


def _is_hole_or_quote(tree):
    return isinstance(tree, (SelectionHole, SelectionQuote))


@contextlib.contextmanager
def _wrap_selection_errors(selection):
    try:
        yield

    except Exception as e:
        raise ValueError("error occurred while building a selection") from e


class Selection:
    def __init__(self, selected_by_path, remainder):
        self.selected_by_path = selected_by_path
        self.remainder = remainder

    @exempt
    def deselect(self):
        def rebuild(subtree):
            if isinstance(subtree, SelectionHole):
                return self.selected_by_path[subtree.path]

            elif isinstance(subtree, SelectionQuote):
                return subtree.quoted

            else:
                return subtree

        with _wrap_selection_errors(self):
            return tree_fn.tree_map(rebuild, self.remainder, is_leaf=_is_hole_or_quote)

    @exempt
    def map(self, fn, *, with_keypath: bool = False, keep_selected: bool = False):
        if with_keypath:
            new_values = {k: fn(k, v) for k, v in self.selected_by_path.items()}

        else:
            new_values = {k: fn(v) for k, v in self.selected_by_path.items()}

        new_selection = Selection(selected_by_path=new_values, remainder=self.remainder)

        if keep_selected:
            return new_selection

        else:
            return new_selection.deselect()

    @exempt
    def map_instance(
        self, fn, *, with_keypath: bool = False, keep_selected: bool = False
    ):
        return self.map(
            config_fn(fn),
            with_keypath=with_keypath,
            keep_selected=keep_selected,
        )

    @exempt
    def replace(
        self, value, *, with_keypath: bool = False, keep_selected: bool = False
    ):
        if with_keypath:
            return self.map(
                lambda _1, _2: value,
                with_keypath=with_keypath,
                keep_selected=keep_selected,
            )

        else:
            return self.map(
                lambda _: value,
                with_keypath=with_keypath,
                keep_selected=keep_selected,
            )

    @exempt
    def update_dict(
        self, update, with_keypath: bool = False, keep_selected: bool = False
    ):
        def wrapper(tree):
            if callable(update):
                updated = update(tree)

            else:
                updated = update

            return type(tree)({**tree, **updated})

        return self.map(wrapper, with_keypath=with_keypath, keep_selected=keep_selected)

    @exempt
    def set_by_path(self, replacer):
        if callable(replacer):
            replacer = {k: replacer(k) for k in self.selected_by_path.keys()}

        return self.map(lambda k, v: replacer[k], with_keypath=True)

    @exempt
    def set_sequence(self, replacements):
        replacer = {}

        for keypath, replacement in zip(self.selected_by_path.keys(), replacements):
            replacer[keypath] = replacement

        return self.set_by_path(replacer)

    @exempt
    def subtrees_where(
        self,
        filter_fn,
        *,
        with_keypath: bool = False,
        absolute_keypath: bool = False,
        innermost: bool = False,
    ):
        def safe_filter_fn(*args):
            result = filter_fn(*args)

            if not isinstance(result, bool):
                raise TypeError("filter_fn must return a boolean value")

            return result

        with _wrap_selection_errors(self):
            if with_keypath or innermost:
                if with_keypath:
                    wrapped_filter_fn = safe_filter_fn

                else:

                    def wrapped_filter_fn(_, s):
                        return safe_filter_fn(s)

                def process_subtree(keypath, leaf_or_subtree):
                    if not innermost:
                        found_here = wrapped_filter_fn(keypath, leaf_or_subtree)

                        if found_here:
                            return True, _SelectionBoundary(leaf_or_subtree)

                    maybe_children = tree_fn.tree_flatten_exactly_one_level(
                        leaf_or_subtree
                    )

                    if maybe_children:
                        keyed_children, treespec = maybe_children
                        new_children = []
                        any_descendant_selected = False

                        for key, child in keyed_children:
                            found_in_child, new_child = process_subtree(
                                keypath + (key,), child
                            )
                            new_children.append(new_child)
                            any_descendant_selected = (
                                any_descendant_selected or found_in_child
                            )

                        if not any_descendant_selected and wrapped_filter_fn(
                            keypath, leaf_or_subtree
                        ):
                            return True, _SelectionBoundary(leaf_or_subtree)

                        elif any_descendant_selected:
                            return True, tree_fn.tree_unflatten(new_children, treespec)

                        else:
                            return False, leaf_or_subtree

                    else:
                        return False, leaf_or_subtree

                def process_selected(keypath, selected_subtree):
                    if absolute_keypath:
                        _, result = process_subtree(keypath, selected_subtree)
                        return result

                    else:
                        _, result = process_subtree((), selected_subtree)
                        return result

                return _build_selection(self.map(process_selected, with_keypath=True))

            else:

                def process_leaf_or_filtered(leaf_or_subtree):
                    if safe_filter_fn(leaf_or_subtree):
                        return _SelectionBoundary(leaf_or_subtree)

                    else:
                        return leaf_or_subtree

                def process_selected(selected_subtree):
                    return tree_fn.tree_map(
                        process_leaf_or_filtered,
                        selected_subtree,
                        is_leaf=safe_filter_fn,
                    )

                return _build_selection(self.map(process_selected))

    @exempt
    def at(self, accessor_fn, multiple: bool | None = None):
        def is_leaf_or_without_children(node):
            result = tree_fn.tree_flatten_exactly_one_level(node)

            if result is None:
                return True

            else:
                children, _ = result

                return not children

        def unwrap(leaf):
            assert isinstance(leaf, _LeafWrapper)

            return leaf.wrapped_leaf

        def process_one(node, multiple=multiple):
            uniquified_copy = tree_fn.tree_map(
                _LeafWrapper, node, is_leaf=is_leaf_or_without_children
            )
            needle_or_needles = accessor_fn(uniquified_copy)

            if multiple:
                needles = needle_or_needles

            else:
                needles = (needle_or_needles,)

            needle_ids = set(id(x) for x in needles)

            leaves_or_needles, treespec = tree_fn.tree_flatten(
                uniquified_copy, is_leaf=lambda t: id(t) in needle_ids
            )
            new_leaves = []
            found_ids = set()

            for leaf_or_needle in leaves_or_needles:
                if id(leaf_or_needle) in needle_ids:
                    if id(leaf_or_needle) in found_ids:
                        raise ValueError("multiple needles found")

                    found_ids.add(id(leaf_or_needle))
                    new_leaves.append(
                        _SelectionBoundary(tree_fn.tree_map(unwrap, leaf_or_needle))
                    )

                else:
                    new_leaves.append(unwrap(leaf_or_needle))

            missing_needles = [
                needle for needle in needles if id(needle) not in found_ids
            ]

            if missing_needles:
                if multiple is None and isinstance(needle_or_needles, Collection):
                    try:
                        result = process_one(node, multiple=True)

                    except ValueError:
                        pass

                    else:
                        return result

                if multiple:
                    raise ValueError("needles not found")

                else:
                    assert len(missing_needles) == 1

                    raise ValueError("needle not found")

            return tree_fn.tree_unflatten(new_leaves, treespec)

        with_boundary = self.map(process_one)

        with _wrap_selection_errors(self):
            return _build_selection(with_boundary)

    @exempt
    def instance(self, instance, innermost: bool = False):
        if not isinstance(instance, str):
            instance_str = import_to_str(instance)

        else:
            instance_str = instance

        def check_instance(subtree):
            if not isinstance(subtree, collections.abc.Mapping):
                return False

            if INIT_KEY in subtree:
                target = subtree[INIT_KEY]

            elif FN_KEY in subtree:
                target = subtree[FN_KEY]

            elif TARGET_KEY in subtree:
                target = subtree[TARGET_KEY]

            else:
                return False

            return target == instance_str

        return self.subtrees_where(check_instance, innermost=innermost)

    @exempt
    def tag(self, tag, innermost: bool = False):
        def check_tag(subtree):
            if not isinstance(subtree, collections.abc.Mapping):
                return False

            if TAG_KEY not in subtree:
                return False

            return subtree[TAG_KEY] == tag

        return self.subtrees_where(check_tag, innermost=innermost)


def _build_selection(tree):
    selected_by_path = {}

    def process(path, leaf):
        if isinstance(leaf, _SelectionBoundary):
            selected_by_path[path] = leaf.selected

            return SelectionHole(path)

        elif _is_hole_or_quote(leaf):
            return SelectionQuote(leaf)

        else:
            return leaf

    remainder = tree_fn.tree_map_with_path(process, tree, is_leaf=_is_hole_or_quote)

    return Selection(selected_by_path=selected_by_path, remainder=remainder)


@exempt
def select(tree):
    return _build_selection(_SelectionBoundary(tree))
