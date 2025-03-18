from slickconf import field

from .conf_lib import test_fn_config

conf = field()
conf.fn_config = test_fn_config(value="test")
