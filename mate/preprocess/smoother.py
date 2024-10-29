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
            avg_term = np.convolve(d, np.ones(self.window_size) / self.window_size, mode='valid')
            avg_data.append(avg_term)

        return np.array(avg_data)

class SavgolSmoother():
    def __init__(self, family):
        self.window_length = 11
        self.polyorder = 2

        if 'window_length' in family:
            self.window_length = family['window_length']
        if 'polyorder' in family:
            self.polyorder = family['polyorder']

    def smoothing(self, arr):
        savgol_data = savgol_filter(x=arr,
                                     window_length=self.window_length,
                                     polyorder=self.polyorder)
        return savgol_data

class ExpMovingAverageSmoother():
    def __init__(self, family):
        self.span = 20

        if 'span' in family:
            self.span = family['span']

    def smoothing(self, arr):
        shape = arr.shape
        df = pd.DataFrame(arr)
        exp_data = df.ewm(span=self.span).mean().to_numpy()

        return exp_data

class LowessSmoother():
    def __init__(self, family):
        self.frac = 0.025

        if 'frac' in family:
            self.frac = family['frac']

    def smoothing(self, arr):
        lowess_data = []
        for data in arr:
            lowess_term = sm.nonparametric.lowess(data, np.arange(len(data)), frac=self.frac)
            lowess_data.append(lowess_term[:, 1])

        return np.array(lowess_data)