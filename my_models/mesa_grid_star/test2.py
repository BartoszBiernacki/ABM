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


# AVG vals form multiple dataframes *********************************************************************************
# df1 = pd.read_csv(filepath_or_buffer="results/first.csv")
# df2 = pd.read_csv(filepath_or_buffer="results/second.csv")
#
#
# average_array_from_frames = np.mean([df1, df2], axis=0)
# print(average_array_from_frames)
# ********************************************************************************************************************
