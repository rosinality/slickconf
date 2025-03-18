from slickconf import imports

with imports():
    from slickconf.test import Container


class TestClass:
    def __init__(self, name):
        self.key = "TestClass"
        self.name = name

    def __repr__(self):
        return f"TestClass(name={self.name})"


def test_fn(name, suffix=""):
    return name + suffix


def test_fn_config(value):
    return Container(str(value) + "!")
