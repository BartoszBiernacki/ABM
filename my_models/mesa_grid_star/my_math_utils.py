import numpy as np


def get_rand_int_from_exp_distribution(mean):
    return round(np.random.exponential(mean))


def get_rand_int_from_triangular_distribution(left, mode, right):
    return round(np.random.triangular(left, mode, right))


