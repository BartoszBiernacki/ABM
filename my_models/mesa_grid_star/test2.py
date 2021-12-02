import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import FormatStrFormatter
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
from numba import prange


sam = random.sample(range(10), 10)
print(sam)
