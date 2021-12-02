import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import itertools
import random
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pathlib import Path


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


def fit_exp_to_peaks(y_data, x_data=None,
                     plot=True, save=True,
                     N=None, beta=None,
                     exposed_period=None, exposed_bins=None, infected_period=None, infected_bins=None,
                     show_details=False):
    if x_data is None:
        x_data = np.arange(start=0, stop=len(y_data))

    # Plot experimental data
    fig = plt.figure()
    ax = fig.add_subplot(111)

    legend1 = 'experimental data'
    ax.plot(x_data, y_data, label=legend1, color="Green", linestyle='dashed', marker='o', markersize=5, zorder=0)

    # Find peaks
    peak_pos_indexes = find_peaks(y_data)[0]
    peak_pos = x_data[peak_pos_indexes]
    peak_height = y_data[peak_pos_indexes]

    # Plot maxima (data used for fitting)
    legend2 = 'maxima'
    ax.scatter(peak_pos, peak_height, label=legend2, zorder=2)

    # Initial guess for the parameters
    initial_guess = [50, 40]
    # Perform the curve-fit
    popt, pcov = curve_fit(unknown_exp_func, peak_pos, peak_height, initial_guess)
    # print(pcov)

    # x values for the fitted function
    x_fit = x_data

    # Plot fitted function
    legend3 = r' $y =  Ae^{{-t/\tau}}$' '\n' r'$A={:.1f}$' '\n' r'$\tau = {:.1f}$'.format(popt[0], popt[1])
    ax.plot(x_fit, unknown_exp_func(x_fit, *popt), 'r', label=legend3, zorder=1)
    ax.set_xlabel('t, days', fontsize=12)
    ax.set_ylabel('Number of exposed', fontsize=12)

    # Show legend and its entries in correct order
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]
    ax.legend(handles, labels)

    if N and beta:
        if not show_details:
            ax.set_title(r"$N = {:1d}, \beta = {:.3f}$".format(N, beta))
        else:
            ax.set_title(r"$N = {:1d}, \beta = {:.3f}$"'\n'
                         r"exposed period = ${:1d}\pm{:1d}$"'\t'
                         r"infected period = ${:1d}\pm{:1d}$".format(N, beta, exposed_period, exposed_bins//2,
                                                                     infected_period, infected_bins//2))
    plt.tight_layout()
    
    if plot:
        plt.show()
        
    if save:
        directory = "results/Exposed_vs_days/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f"{directory}"
                    f"Beta={beta}"
                    f"_Exposed_duration={exposed_period}("f"{exposed_bins//2})_days"
                    f"_Infection_duration={infected_period}("f"{infected_bins//2})_days"
                    f".pdf")

    return popt[1]  # returns tau
