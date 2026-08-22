import inspect
from copy import deepcopy
from functools import partial
from pathlib import Path


class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj: object) -> None:
        for key in dir(obj):
            if key == "__class__":
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                instance = value()
                setattr(obj, key, instance)
                BaseConfig.init_member_classes(instance)
            elif isinstance(value, (dict, list, set)):
                setattr(obj, key, deepcopy(value))