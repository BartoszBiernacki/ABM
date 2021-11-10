import numpy as np
from numba import jit, njit, int32, float32, float64, typeof
from numba.experimental import jitclass
import itertools


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

