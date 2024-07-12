from slickconf import field, tag, call, config_fn
from slickconf.test import Linear, Sequential

from .eager_call import create_linear_export

conf = field()
conf.module = call[create_linear_export](3, 5)