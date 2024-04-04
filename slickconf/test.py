class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(repr, self.layers))})"
