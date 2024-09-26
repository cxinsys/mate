from mate.preprocess import Discretizer, ShiftDiscretizer, InterpDiscretizer, TagDiscretizer
from mate.preprocess import MovingAvgSmoother, SavgolSmoother, LowessSmoother, ExpMovingAverageSmoother

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


        raise ValueError(f"{_method} is not a supported discretizer.")

class SmootherFactory:
    @staticmethod
    def create(method, *args, **kwargs):
        _method = method.lower()

        if 'moving_avg' in _method:
            return MovingAvgSmoother(*args, **kwargs)
        elif 'savgol' in _method:
            return SavgolSmoother(*args, **kwargs)
        elif 'moving_exp' in _method:
            return ExpMovingAverageSmoother(*args, **kwargs)
        elif 'loess' or 'lowess' in _method:
            return LowessSmoother(*args, **kwargs)

        raise ValueError(f'{_method} is not supported smoother.')