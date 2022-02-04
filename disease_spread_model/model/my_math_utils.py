import numpy as np
from numba import njit
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


def find_best_x_shift_to_match_plots(y1_reference, y2, y2_start, y2_end):
    """
    Returns index of elem from which data y2[start: stop] best match any slice of the same length of y1.
    Also returns fit error (SSE).
    """
    y1_reference = np.array(y1_reference)
    y2 = np.array(y2[y2_start: y2_end+1])
    
    smallest_difference = 1e9
    y2_length = y2_end - y2_start + 1
    shift = 0
    
    for i in range(len(y1_reference) - len(y2) + 1):
        y1_subset = y1_reference[i: i+y2_length]
        
        difference = np.sum((y2 - y1_subset)**2)
        
        if difference < smallest_difference:
            smallest_difference = difference
            shift = i
            
    shift = y2_start - shift
    
    return shift, smallest_difference