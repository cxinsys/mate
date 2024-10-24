from mate.preprocess import Discretizer, ShiftDiscretizer, InterpDiscretizer,\
    TagDiscretizer, FixedWidthDiscretizer, QuantileDiscretizer, KmeansDiscretizer, LogDiscretizer
from mate.preprocess import MovingAvgSmoother, SavgolSmoother, LowessSmoother, ExpMovingAverageSmoother

class DiscretizerFactory:
    @staticmethod
    def create(method, binningfamily: dict = None, *args, **kwargs):
        if not method:
            return None
        _method = method.lower()

        if _method == "FSBW":
            return Discretizer(*args, **kwargs)
        elif _method == "FSBW-L":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "FSBW-R":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "FSBW-B":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "FSBW-I":
            return InterpDiscretizer(*args, **kwargs)
        elif _method == "FSBW-T":
            return TagDiscretizer(*args, **kwargs)
        elif _method == "FSBN":
            return FixedWidthDiscretizer(family=binningfamily, *args, **kwargs)
        elif _method == "FSBQ":
            return QuantileDiscretizer(family=binningfamily, *args, **kwargs)
        elif _method == "K-means":
            return KmeansDiscretizer(family=binningfamily, *args, **kwargs)
        elif "log" in _method:
            return LogDiscretizer(family=binningfamily, *args, **kwargs)

        raise ValueError(f"{_method} is not a supported discretizer.")

class SmootherFactory:
    @staticmethod
    def create(smoothfamily: dict = None):
        if not smoothfamily or smoothfamily == 'None':
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