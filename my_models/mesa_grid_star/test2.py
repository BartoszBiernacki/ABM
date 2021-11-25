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


@njit(cache=True)
def func(neighbouring_neighbourhoods_grouped_by_neighbourhood_id, array_to_fill):
    # result[neighbourhood_id][num_of_household_in_that_neighbourhood][week] --> neighbouring neighbourhood to visit

    total_num_of_neighbourhoods, num_of_neighbouring_neighbourhoods = \
        neighbouring_neighbourhoods_grouped_by_neighbourhood_id.shape

    if num_of_neighbouring_neighbourhoods:  # do sth if there is at least one neighbour
        total_num_of_neighbourhoods, num_of_households_in_one_neighbourhood, num_of_weeks_to_simulate = np.shape(
            array_to_fill)
        num_of_needed_cycles = num_of_weeks_to_simulate // num_of_neighbouring_neighbourhoods + 1
        for neighbourhood_id in range(total_num_of_neighbourhoods):
            for household in range(num_of_households_in_one_neighbourhood):
                covered_weeks = 0
                for cycle in range(num_of_needed_cycles):
                    ran = np.random.permutation(neighbouring_neighbourhoods_grouped_by_neighbourhood_id[
                                                   neighbourhood_id])
                    for i in range(num_of_neighbouring_neighbourhoods):
                        array_to_fill[neighbourhood_id][household][covered_weeks] = ran[i]
                        covered_weeks += 1
                        if covered_weeks == num_of_weeks_to_simulate:
                            break
        return array_to_fill


neighbouring_neighbourhoods_grouped_by_neighbourhood_id = np.array([[1, 4],
                                                                    [0, 2],
                                                                    [1, 3],
                                                                    [2, 4],
                                                                    [3, 0]])


array_to_fill = np.empty((5, 8, 10), dtype=np.int8)

with cProfile.Profile() as pr:
    order = func(neighbouring_neighbourhoods_grouped_by_neighbourhood_id
                 =neighbouring_neighbourhoods_grouped_by_neighbourhood_id,
                 array_to_fill=array_to_fill)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(10)

print(order)

a = np.array([0, 1, 2, 3, 4, 5, 6, 4, 11])
ind = [2, 7]
print(a[ind])
