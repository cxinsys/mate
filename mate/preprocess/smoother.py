import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import savgol_filter

class MovingAvgSmoother():
    def __init__(self,
                 window_size,
                 ):

        self.window_size = window_size
    def smoothing(self, arr):
        avg_data = []
        for d in arr:
            avg_term = np.convolve(d, np.ones(self.window_size) / self.window_size, mode='valid')
            avg_data.append(avg_term)

        return np.array(avg_data)

class SavgolSmoother():
    def __init__(self,
                 window_length=None,
                 polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def smoothing(self, arr):
        return savgol_filter(x=arr,
                             window_length=self.window_length,
                             polyorder=self.polyorder)

class ExpMovingAverageSmoother():
    def __init__(self, span=20):
        self.span = span

    def smoothing(self, arr):
        df = pd.DataFrame(arr)
        return df.ewm(span=self.span).mean()

class LowessSmoother():
    def __init__(self, frac=0.025):
        self.frac = frac

    def smoothing(self, arr):
        lowess_data = []
        for data in arr:
            lowess_term = sm.nonparametric.lowess(data, np.arange(len(data)), frac=self.frac)
            lowess_data.append(lowess_term[:, 1])

        return np.array(lowess_data)