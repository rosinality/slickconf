import pytest
from pydantic import ValidationError

from slickconf import load_config, instantiate, select, Config, Instance
from slickconf.test import (
    Linear,
    Sequential,
    is_serializable,
    NamedContainer,
    BetterContainer,
    Container,
)


@pytest.fixture
def conf(assets_dir):
    return load_config(assets_dir / "autoconfig_conf.py")


def test_basic_construct(conf):
    assert is_serializable(conf.module)

    module = instantiate(conf.module)

    assert isinstance(module, Sequential)
    assert len(module.layers) == 2
    assert isinstance(module.layers[0], Linear)
    assert isinstance(module.layers[1], Linear)
    assert module.layers[0].in_features == 2
    assert module.layers[0].out_features == 1
    assert module.layers[1].in_features == 1
    assert module.layers[1].out_features == 2


def test_singleton(conf):
    assert is_serializable(conf.module_single)

    module = instantiate(conf.module_single)

    assert module.layers[0] is module.layers[1]


def test_partial(conf):
    assert is_serializable(conf.partial_fn)

    partial_fn = instantiate(conf.partial_fn)

    assert partial_fn(arg2="test2") == "test1 test2 test3"


def test_args_binding(conf):
    assert is_serializable(conf.fns)

    assert instantiate(conf.fns.args) == ("a", ("c",), "b")
    assert instantiate(conf.fns.plain) == ("a", "b")
    assert instantiate(conf.fns.position_only) == ("a", "b")
    assert instantiate(conf.fns.keyword_only) == ("a", "b")
    assert instantiate(conf.fns.kwargs) == ("a", (("b", "b"), ("c", "c")))

    with pytest.raises(TypeError):
        instantiate(conf.fns.keyword_only, "i", "j")

    with pytest.raises(TypeError):
        instantiate(conf.fns.position_only, a="i")


def test_deepcopy(conf):
    assert is_serializable(conf.module_copy)

    module = instantiate(conf.module_copy)

    assert module.layers[0] is module.layers[1]
    assert module.layers[0] is not module.layers[2]


def test_external(conf):
    assert is_serializable(conf.external)

    test_class = instantiate(conf.external.test_class)

    assert test_class.key == "TestClass"
    assert test_class.name == "test_class"
    assert instantiate(conf.external.test_fn) == "test_fn"
    assert instantiate(conf.external.test_fn_partial)("test") == "testsuffix"


def test_ref(assets_dir):
    conf = load_config(
        assets_dir / "autoconfig_conf.py", overrides=("hparams.batch_size=64",)
    )

    assert is_serializable(conf.training)

    assert conf.training.max_iter == 62500


def test_lazy(assets_dir):
    conf = load_config(
        assets_dir / "autoconfig_conf.py",
        overrides=("hparams.name='test'",),
    )

    assert is_serializable(conf.lazy_class)

    assert instantiate(conf.lazy_class).value == "test!"


def test_select(conf):
    assert is_serializable(conf.select)

    selection = select(conf.select)

    mapped = instantiate(selection.instance(NamedContainer).map(lambda x: "SELECTED"))
    assert mapped.layers[0] == "SELECTED"
    assert mapped.layers[1] == "SELECTED"

    mapped = instantiate(
        selection.instance(NamedContainer).map_instance(
            lambda x: BetterContainer(x.name + "!", x.value)
        )
    )

    assert mapped.layers[0].name == "test!"
    assert tuple(mapped.layers[0].value) == (0, 1, 2)
    assert mapped.layers[1].name == "test2!"
    assert tuple(mapped.layers[1].value) == (0, 1, 2)

    mapped = instantiate(
        selection.instance(NamedContainer).update_dict({"name": "Updated"})
    )

    assert mapped.layers[0].name == "Updated"
    assert mapped.layers[1].name == "Updated"


def test_select_tag(conf):
    assert is_serializable(conf.select_tag)

    selection = select(conf.select_tag)

    mapped = instantiate(selection.tag("layer").replace("placeholder"))
    assert mapped.layers[0] == "placeholder"

    original = instantiate(conf.select_tag)
    assert isinstance(original.layers[0], Linear)
    assert original.layers[0].in_features == 2
    assert original.layers[0].out_features == 3


class TypedFn(Config):
    typed_fn: Instance


def test_typed_fn(assets_dir):
    with pytest.raises(ValidationError):
        load_config(assets_dir / "autoconfig_type_error.py", TypedFn)


def test_config_fn(conf):
    fn_config = instantiate(conf.fn_config("test"))

    assert isinstance(fn_config, Container)
    assert fn_config.value == "test!"
