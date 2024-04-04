class AnyConfig(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = AnyConfig(**value)
            self[key] = value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{item}'"
            ) from e
