from mate.preprocess import Discretizer, ShiftDiscretizer, InterpDiscretizer,\
    TagDiscretizer, FixedWidthDiscretizer, QuantileDiscretizer, KmeansDiscretizer, LogDiscretizer
from mate.preprocess import MovingAvgSmoother, SavgolSmoother, LowessSmoother, ExpMovingAverageSmoother

class DiscretizerFactory:
    @staticmethod
    def create(binning_method, binning_family: dict = None, *args, **kwargs):
        if not binning_method:
            return None
        _method = binning_method.lower()

        if _method == "fsbw":
            return Discretizer(*args, **kwargs)
        elif _method == "fsbw-l":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "fsbw-r":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "fsbw-b":
            return ShiftDiscretizer(_method, *args, **kwargs)
        elif _method == "fsbw-i":
            return InterpDiscretizer(*args, **kwargs)
        elif _method == "fsbw-t":
            return TagDiscretizer(*args, **kwargs)
        elif _method == "fsbn":
            return FixedWidthDiscretizer(family=binning_family, *args, **kwargs)
        elif _method == "fsbq":
            return QuantileDiscretizer(family=binning_family, *args, **kwargs)
        elif _method == "K-means":
            return KmeansDiscretizer(family=binning_family, *args, **kwargs)
        elif "log" in _method:
            return LogDiscretizer(family=binning_family, *args, **kwargs)

        raise ValueError(f"{_method} is not a supported discretizer.")

class SmootherFactory:
    @staticmethod
    def create(smoothing_opt: dict = None):
        if not smoothing_opt or smoothing_opt == 'None':
            return None
        _method = smoothing_opt['method'].lower()

        if 'mov' in _method:
            return MovingAvgSmoother(smoothing_opt)
        elif 'savgol' in _method:
            return SavgolSmoother(smoothing_opt)
        elif 'exp' in _method:
            return ExpMovingAverageSmoother(smoothing_opt)
        elif 'loess' or 'lowess' in _method:
            return LowessSmoother(smoothing_opt)

        raise ValueError(f'{_method} is not supported smoother.')