from mate.preprocess import Discretizer
from mate.preprocess import ShiftDiscretizer
from mate.preprocess import InterpDiscretizer
from mate.preprocess import TagDiscretizer

class DiscretizerFactory:
    @staticmethod
    def create(method, *args, **kwargs):
        _method = method.lower()

        # print(f"Method designated: {_method.upper()}")

        if "default" in _method:
            return Discretizer(*args, **kwargs)
        elif "shift_left" in _method:
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif "shift_right" in _method:
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif "shift_both" in _method:
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif "interpolation" in _method:
            return InterpDiscretizer(*args, **kwargs)
        elif "tag" in _method:
            return TagDiscretizer(*args, **kwargs)


        raise ValueError(f"{_method} is not a supported solver.")