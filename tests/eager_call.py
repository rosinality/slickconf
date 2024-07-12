from slickconf import field, tag, call, config_fn
from slickconf.test import Linear, Sequential


def sequential(module):
    return Sequential(module)


def create_linear(in_features, out_features):
    layers = []
    
    for _ in call[range](3):
        call[layers.append](Linear(in_features, out_features))
    
    return call[sequential](layers)


conf = field()
conf.module = Sequential(call[create_linear](3, 5), call[create_linear](5, 1))


@config_fn
def create_linear_export(in_features, out_features):
    layers = []
    
    for _ in call[range](3):
        call[layers.append](Linear(in_features, out_features))
    
    return call[sequential](layers)