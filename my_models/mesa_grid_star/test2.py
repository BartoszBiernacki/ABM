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
from my_models.mesa_grid_star.my_model_data_analysis import find_beta_for_all_voivodeships
from text_processing import *
from numba import prange
from data_visualisation import *
import matplotlib

from pathlib import Path
import pathlib

from real_data import RealData
# from constants import *
from avg_results import get_single_results
from config import Config


if __name__ == '__main__':
    voivodeship = Config.voivodeship
    percent_of_touched_counties = Config.percent_of_touched_counties
    days_to_fit_death_toll = Config.days_to_fit_death_toll
    # ----------------------------------------------------------------------------------------------------------
    directory = 'results/Lubelskie/Runs=5___Grid_size=(29, 29)___N=835___Customers_in_household=3' \
                '___Infected_cashiers_at_start=29___Infect_housemates_boolean=0/' \
                'raw data/'
    # ----------------------------------------------------------------------------------------------------------
    real_data_obj = RealData(customers_in_household=3)
    real_general_data = real_data_obj.get_real_general_data()
    real_infected_toll = real_data_obj.get_real_infected_toll()
    real_death_toll = real_data_obj.get_real_death_toll()
    real_death_toll_shifted_by_num_of_deaths = \
        real_data_obj.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(
            starting_day=10, minimum_deaths=5)
    starting_day = real_data_obj.get_starting_days_for_voivodeships_based_on_district_deaths(
        percent_of_touched_counties=20,
        ignore_healthy_counties=True)
    # ----------------------------------------------------------------------------------------------------------

from test_small import RealData


class RealVisualisation(object):
    def __init__(self):
        pass
