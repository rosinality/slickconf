from slickconf import field, tag
from slickconf.test import Linear, Sequential

conf = field()
conf.module = Sequential(Linear(2, tag("hidden")), Linear(tag("hidden"), 1))
conf.sub_module = Sequential(
    Linear(in_features=tag("input"), out_features=16),
    Linear(in_features=16, out_features=tag("input")),
)
