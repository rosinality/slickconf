from functools import partial

from slickconf import field

from slickconf.test import blank_fn, nested_dict


conf = field(
    target=nested_dict(
        id=1,
        name="test",
        fields={"default": partial(blank_fn, 1)},
    )
)
