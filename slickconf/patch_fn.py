import re
from slickconf.builder import exempt, import_to_str
from slickconf.container import NodeDict
from slickconf.constants import SIGNATURE_KEY, ARGS_KEY, INIT_KEY


def _op(type, **kwargs):
    return {"type": type, **kwargs}


class Select:
    def __init__(self):
        self._chains = [_op("select")]

    @exempt
    def chain(self):
        return self._chains

    @exempt
    def at(self, key):
        self._chains.append(_op("at", key=key))

        return self

    @exempt
    def instance(self, instance_str):
        if not isinstance(instance_str, str):
            instance_str = import_to_str(instance_str)

        self._chains.append(_op("instance", instance=instance_str))

        return self

    @exempt
    def map_instance(self, target, *args, **kwargs):
        if not isinstance(target, str):
            target_str = import_to_str(target)
            target = NodeDict.build("__init", target_str, target, args, kwargs)
            signature = target.get(SIGNATURE_KEY, None)
            args = target.get(ARGS_KEY, ())
            kwargs = {
                k: v
                for k, v in target.items()
                if k not in {INIT_KEY, SIGNATURE_KEY, ARGS_KEY}
            }

        else:
            target_str = target
            signature = None

        self._chains.append(
            _op(
                "map_instance",
                target=target_str,
                signature=signature,
                args=args,
                kwargs=kwargs,
            )
        )

        return self

    @exempt
    def update_dict(self, value):
        self._chains.append(_op("update_dict", value=value))

        return self

    @exempt
    def set_sequence(self, replace, start=0, step=1):
        self._chains.append(
            _op("set_sequence", replace=replace, start=start, step=step)
        )

        return self


@exempt
def select():
    return Select()


@exempt
def setattr(path, value):
    return _op("setattr", path=path, value=value)
