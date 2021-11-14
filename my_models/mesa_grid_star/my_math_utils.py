import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit, int32, float32, float64, typeof
from numba.experimental import jitclass
import itertools
import random
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def get_rand_int_from_exp_distribution(mean):
    return np.round(np.random.exponential(mean)).astype(int)


def get_rand_int_from_triangular_distribution(left, mode, right):
    return np.round(np.random.triangular(left, mode, right)).astype(int)


def get_rand_n_ints_from_triangular_distribution(left, mode, right, n):
    vals = np.random.triangular(left, mode, right, n).round()
    int_vals = np.array(vals, dtype=np.int32)
    return int_vals


def get_needed_number_of_processes(num_of_cores, total_num_of_iterations):
    if total_num_of_iterations < num_of_cores:
        return total_num_of_iterations
    else:
        return num_of_cores


def group_tuples_by_start(list_of_tuples, start_length):
    result = {}
    tuples_starts = [[item for i, item in enumerate(tup) if i < start_length] for tup in list_of_tuples]
    tuples_starts.sort()
    unique_tuples_starts = list(tuples_starts for tuples_starts, _ in itertools.groupby(tuples_starts))

    for unique_tuple_start in unique_tuples_starts:
        tuples_grouped_by_start = []
        for tup in list_of_tuples:
            tuple_start = tup[:start_length]
            if list(tuple_start) == unique_tuple_start:
                tuples_grouped_by_start.append(tup)
        result[tuple(unique_tuple_start)] = tuples_grouped_by_start
    return result


@njit
def get_number_of_cell_neighbours(y, x):
    if x == y == 1:
        return 0
    elif x*y == 2:
        return 1
    elif (y == 1 and x >= 3) or (x == 1 and y >= 3) or x == y == 2:
        return 2
    elif x*y >= 6 and (x == 2 or y == 2):
        return 3
    else:
        return 4


def create_array_of_shopping_days_for_each_household(total_num_of_households):
    result = np.empty((total_num_of_households, 2), dtype=int)
    for household in range(total_num_of_households):
        result[household] = random.sample(range(0, 7), 2)
    return result


# Fitting function
def func(t, A, tau):
    return A * np.exp(-t/tau)


def fit_exp_to_peaks(y_data, x_data=None, plot=True):
    if x_data is None:
        x_data = np.arange(start=0, stop=len(y_data))

    if plot:
        plt.plot(x_data, y_data, label='experimental-data')

    # Find peaks
    peak_pos_indexes = find_peaks(y_data)[0]
    peak_pos = x_data[peak_pos_indexes]
    peak_height = y_data[peak_pos_indexes]

    if plot:
        plt.scatter(peak_pos, peak_height, label='maxima')

    # Initial guess for the parameters
    initial_guess = [50, 10]

    # Perform the curve-fit
    popt, pcov = curve_fit(func, peak_pos, peak_height, initial_guess)
    print(popt)
    # print(pcov)

    # x values for the fitted function
    x_fit = x_data

    # Plot the fitted function
    if plot:
        plt.plot(x_fit, func(x_fit, *popt), 'r', label='fit params: a=%5.3f, b=%5.3f' % tuple(popt))

    plt.show()

    return popt[1]




