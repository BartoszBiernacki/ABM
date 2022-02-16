import numpy as np
import random
import math
from numba import njit
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


def get_S_and_exponents_for_sym_hist(bins: int) -> '(S, exponents)':
    if not bins % 2:
        raise ValueError(f"Number of bins must be odd to create histogram with single peak, while {bins} were given!")

    n = bins // 2
    S = 2 * np.sum(2 ** i for i in range(n))
    S += 2 ** n

    exponents = np.array([], dtype=int)
    first = [i for i in range(bins // 2)]
    exponents = np.append(exponents, first)
    exponents = np.append(exponents, bins // 2)
    exponents = np.append(exponents, list(reversed(first)))

    return S, exponents


@njit(cache=True)
def int_from_hist(mean: int, bins: int, S: int, exponents: np.ndarray):
    norm_sum = 0
    r = random.random()
    for i in range(bins):
        if norm_sum <= r < norm_sum + (2 ** exponents[i]) / S:
            return i + mean - bins//2
        else:
            norm_sum += (2 ** exponents[i]) / S


def get_needed_number_of_processes(num_of_cores, total_num_of_iterations):
    if total_num_of_iterations < num_of_cores:
        return total_num_of_iterations
    else:
        return num_of_cores


@njit(cache=True)
def nearest_neighbours(y, x):
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


# Signature of a function to fit
def unknown_exp_func(t, A, tau):
    return A * np.exp(-t/tau)


def calc_tau(x_data, y_data):
    # Find peaks
    peak_pos_indexes = find_peaks(y_data)[0]
    peak_pos = x_data[peak_pos_indexes]
    peak_height = y_data[peak_pos_indexes]

    # Initial guess for the parameters
    initial_guess = [np.max(y_data), 50]
    # Perform  curve-fit
    try:
        popt, pcov = curve_fit(unknown_exp_func, peak_pos, peak_height, initial_guess)
    except ValueError('fit_exp_to_peaks function cannot perform fitting'):
        raise

    return popt[0], popt[1]  # returns (A, tau)


def calc_exec_time(grid_size, N, household_size, max_steps, iterations, betas, mortalities, visibilities):
    coef = (2100 / (20*20*1000*400*20*4*4*5)) / 60
    
    runs = iterations*betas*mortalities*visibilities
    
    household_factor = 0.4*household_size + 0.6
    coef = coef*household_factor
    
    val = int(grid_size[0]*grid_size[1]) * int(N*max_steps*runs)
    
    calc_time_minutes = val * coef
    if calc_time_minutes > 60:
        hours = calc_time_minutes // 60
        minutes = round(calc_time_minutes % 60, 1)
    else:
        hours = 0
        minutes = round(calc_time_minutes, 1)
        
    print(f'It will take {hours} hours and {minutes} minutes to evaluate {runs} simulations')


def sort_df_indices_by_col(df, column):
    """
    Returns dict in which keys are indices of dataframe and val is pos of that index
    in df sorted by specified column
    """
    result = {}
    df = df.sort_values(by=column)
    
    for i, df_index in enumerate(df.index):
        result[df_index] = i
    
    return result


def sort_dict_by_values(dictionary):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}


def window_derivative(y, half_win_size):
    x = np.arange(len(y))
    y_prime = np.zeros_like(y)
    for i in range(len(y)):
        left = max(0, i - half_win_size)
        right = min(len(y) - 1, i + half_win_size)
        dy = y[right] - y[left]
        dx = x[right] - x[left]
        y_prime[i] = dy / dx
    
    return y_prime


def complete_missing_data(values: np.ndarray):
    """
    Completes nans in array of floats by interpolation.
    Example: [0, 1, 2, np.NaN, 4, np.NaN, np.NaN, 7] --> [0, 1, 2, 3, 4, 5, 6, 7]
    
    Returns completed copy of passed array.
    
    Note: if 'values' ends with np.NaN, then last sequence of nans
        is not modified e.g. [0, np.NaN, 2, np.NaN] --> [0, 1, 2, np.NaN].
    
    :param values: numpy array of floats with missing entries.
    :type values: np.ndarray
    """
    
    # copy values (not modify passed array)
    completed = np.copy(values)
    
    # iterate over all values in array
    for i, val in enumerate(completed):
        # is first value NaN?
        if math.isnan(val) and i == 0:
            completed[i] = 0
        # is any other value NaN?
        elif math.isnan(val):
            # iterate over array items right after NaN appeared
            # to find out where NaN sequence ends
            for j in range(i + 1, len(completed)):
                # if NaN sequence ended, interpolate and fill 'completed'
                if not math.isnan(completed[j]):
                    min_val = completed[i - 1]
                    max_val = completed[j]
                    
                    delta_y = max_val - min_val
                    delta_x = j - i + 1
                    dy = delta_y / delta_x
                    
                    complementary_indices = np.arange(i, j)
                    complementary_values = [min_val + dy * step for step in range(1, delta_x)]
                    
                    # print('delta_y', delta_y)
                    # print('delta_x', delta_x)
                    # print('dy', dy)
                    # print('min value', min_val)
                    # print('complementary indices', complementary_indices)
                    # print('complementary values', complementary_values)
                    # print()
                    
                    completed[complementary_indices] = complementary_values

                    break
    
    return completed


def get_indices_of_missing_data(data: np.ndarray):
    """
    Returns indices of nans in data array of floats by interpolation.
    Output is 2D list where each entry is an array of uninterrupted sequence of nans.
    
    Example: [0, 1, 2, np.NaN, 4, np.NaN, np.NaN, 7] --> [[3], [5, 6]]
    
    Note: if 'data' ends with np.NaN, then last sequence of nans
        is not returned e.g. [0, np.NaN, 2, 3, np.NaN] --> [[1]].
    
    
    Implementation detail: I will iterate over data elem by elem if they
        are not nans. If I get nan I will start iterate from its index until I get
        first not nan value. From here I get uninterrupted sequence of nans.
        To not duplicate sequences I fill nans with zeros and continue top level
        iteration. To not modify data function operates on copy of passed data.


    :param data: numpy array of floats with missing entries.
    :type data: np.ndarray
    :return: list of np.ndarray, each np.ndarray contains uninterrupted sequence
        of indices nans
    :rtype: list
    """
    
    # It will be filled and returned
    missing_indices = []
    
    completed = np.copy(data)
    for i, val in enumerate(completed):
        # is NaN?
        if math.isnan(val):
            # iterate over array items right after NaN appeared
            # to find out where NaN sequence ends
            for j in range(i + 1, len(completed)):
                # If NaN sequence ended fill 'completed' and append
                # found sequence to resulting list
                if not math.isnan(completed[j]):

                    complementary_indices = np.arange(i, j)
                    missing_indices.append(complementary_indices)

                    completed[complementary_indices] = np.zeros_like(complementary_indices)
                    break
                    
    return missing_indices


def slope_from_linear_fit(data, half_win_size):
    from scipy.stats import linregress

    slope = np.zeros_like(data)

    for i in range(len(data)):
        left = max(0, i - half_win_size)
        right = min(len(data) - 1, i + half_win_size)
    
        y = np.array(data[left: right + 1]).astype(float)
        x = np.arange(len(y))
    
        linefit = linregress(x, y)
        slope[i] = linefit.slope
    return slope


