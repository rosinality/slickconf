from slickconf import field, imports

with imports():
    from slickconf.test import typed_fn

conf = field()
conf.typed_fn = typed_fn(1, "2")
