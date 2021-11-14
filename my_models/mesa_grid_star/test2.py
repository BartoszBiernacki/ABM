import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import cProfile
import pstats
from collections import OrderedDict
import os
from multiprocessing import Process
import pandas as pd
import itertools
import sys

from my_math_utils import *

a = np.array([1, 2, 3, 4])
b = np.array([11, 22, 33, 44])

C = np.array([a, b])


x = np.linspace(start=0, stop=30, num=2000)
y = 0.33*np.exp(-x/4) * np.sin(x)


# Experimental x and y data points
xData = np.array([1, 2, 3, 4, 5])
yData = np.array([1, 9, 50, 300, 1500])

# fit_exp_to_peaks(data=y)
# fit_exp_to_peaks(y_data=y)
fit_exp_to_peaks(x_data=x, y_data=y)


# AVG vals form multiple dataframes *********************************************************************************
# df1 = pd.read_csv(filepath_or_buffer="results/first.csv")
# df2 = pd.read_csv(filepath_or_buffer="results/second.csv")
#
#
# average_array_from_frames = np.mean([df1, df2], axis=0)
# print(average_array_from_frames)
# ********************************************************************************************************************
