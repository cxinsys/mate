import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import savgol_filter

class MovingAvgSmoother():
    def __init__(self, family):
        self.window_size = 11

        if 'window_size' in family:
            self.window_size = family['window_size']

    def smoothing(self, arr):
        avg_data = []
        for d in arr:
            avg_term = np.convolve(d.squeeze(1), np.ones(self.window_size) / self.window_size, mode='valid')
            avg_data.append(avg_term)

        return np.expand_dims(np.array(avg_data), -1)

class SavgolSmoother():
    def __init__(self, family):
        self.window_length = 11
        self.polyorder = 2

        if 'window_length' in family:
            self.window_length = family['window_length']
        if 'polyorder' in family:
            self.polyorder = family['polyorder']

    def smoothing(self, arr):
        savgol_data = savgol_filter(x=arr.squeeze(-1),
                                     window_length=self.window_length,
                                     polyorder=self.polyorder)
        return np.expand_dims(savgol_data, -1)

class ExpMovingAverageSmoother():
    def __init__(self, family):
        self.span = 20

        if 'span' in family:
            self.span = family['span']

    def smoothing(self, arr):
        shape = arr.shape
        df = pd.DataFrame(arr.squeeze(-1))
        exp_data = df.ewm(span=self.span).mean().to_numpy()

        return np.expand_dims(exp_data, -1)

class LowessSmoother():
    def __init__(self, family):
        self.frac = 0.025

        if 'frac' in family:
            self.frac = family['frac']

    def smoothing(self, arr):
        lowess_data = []
        for data in arr:
            lowess_term = sm.nonparametric.lowess(data.squeeze(1), np.arange(len(data)), frac=self.frac)
            lowess_data.append(lowess_term[:, 1])

        return np.expand_dims(np.array(lowess_data), -1)