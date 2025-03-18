from copy import deepcopy
from functools import partial

from slickconf import field, external, lazy, tag, config_fn
from slickconf.test import (
    Linear,
    Sequential,
    simple_fn,
    fn_args,
    fn_plain,
    fn_position_only,
    fn_keyword_only,
    fn_kwargs,
    NamedContainer,
    Container,
)

from .conf_lib import TestClass, test_fn
from .config_fn import test_fn_config

conf = field()
conf.module = Sequential(Linear(2, 1), Linear(1, 2))

linear = Linear(2, 1)
conf.module_single = Sequential(linear, linear)

conf.partial_fn = partial(simple_fn, "test1", arg3="test3")

conf.fns = field()
conf.fns.args = fn_args("a", "c", b="b")
conf.fns.plain = fn_plain("a", b="b")
conf.fns.position_only = fn_position_only("a", "b")
conf.fns.keyword_only = fn_keyword_only("a", b="b")
conf.fns.kwargs = fn_kwargs("a", b="b", c="c")

linear = Linear(2, 2)
conf.module_copy = Sequential(linear, linear, deepcopy(linear))

conf.external = field()
conf.external.test_class = external[TestClass](name="test_class")
conf.external.test_fn = external[test_fn](name="test_fn")
conf.external.test_fn_partial = partial(external[test_fn], suffix="suffix")

hparams = field(
    name="base",
    max_length=2048,
    batch_size=16,
    max_tokens=8192 * (10**6),
)

conf.hparams = hparams

conf.training = field(
    max_iter=hparams.ref.max_tokens // hparams.ref.max_length // hparams.ref.batch_size
)


conf.lazy_class = lazy[test_fn_config](value=hparams.ref.name)

conf.select = Sequential(
    NamedContainer("test", tuple(range(3))), NamedContainer("test2", tuple(range(3)))
)

conf.select_tag = Sequential(tag("layer", Linear(2, 3)))


@config_fn
def fn_config(value):
    return Container(str(value) + "!")


conf.fn_config = fn_config
