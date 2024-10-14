from mate.preprocess import Discretizer, ShiftDiscretizer, InterpDiscretizer,\
    TagDiscretizer, FixedWidthDiscretizer, QuantileDiscretizer, KmeansDiscretizer, LogDiscretizer
from mate.preprocess import MovingAvgSmoother, SavgolSmoother, LowessSmoother, ExpMovingAverageSmoother

class DiscretizerFactory:
    @staticmethod
    def create(method, binningfamily: dict = None, *args, **kwargs):
        if not method:
            return None
        _method = method.lower()

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
        elif "fix" in _method:
            return FixedWidthDiscretizer(family=binningfamily, *args, **kwargs)
        elif "quantile" in _method:
            return QuantileDiscretizer(family=binningfamily, *args, **kwargs)
        elif "kmeans" in _method:
            return KmeansDiscretizer(family=binningfamily, *args, **kwargs)
        elif "log" in _method:
            return LogDiscretizer(family=binningfamily, *args, **kwargs)

        raise ValueError(f"{_method} is not a supported discretizer.")

class SmootherFactory:
    @staticmethod
    def create(smoothfamily: dict = None):
        if not smoothfamily or not isinstance(smoothfamily, dict):
            return None
        _method = smoothfamily['method'].lower()

        if 'mov' in _method:
            return MovingAvgSmoother(smoothfamily)
        elif 'savgol' in _method:
            return SavgolSmoother(smoothfamily)
        elif 'exp' in _method:
            return ExpMovingAverageSmoother(smoothfamily)
        elif 'loess' or 'lowess' in _method:
            return LowessSmoother(smoothfamily)

        raise ValueError(f'{_method} is not supported smoother.')