from slickconf import config_fn
from slickconf.test import Container


@config_fn
def test_fn_config(value):
    return Container(str(value) + "!")
