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


def test_drawing_numbers_from_my_hist_distribution(means, bins_array, show, save):
	for mean in means:
		for bins in bins_array:
			S, exponents = get_S_and_exponents_for_sym_hist(bins=bins)
			sample_size = 100000
			mean = mean
			vals = np.zeros(sample_size)
	
			with cProfile.Profile() as pr:
				for i in range(sample_size):
					vals[i] = int_from_hist(mean=mean, bins=bins, S=S, exponents=exponents)
			stats = pstats.Stats(pr)
			stats.sort_stats(pstats.SortKey.TIME)
			stats.print_stats(10)
	
			labels, counts = np.unique(vals, return_counts=True)
			plt.bar(labels, counts, align='center')
			plt.gca().set_xticks(labels)
			
			if show:
				plt.show()
			
			if save:
				directory = "results/tests/histogram_drawing_tests/"
				Path(directory).mkdir(parents=True, exist_ok=True)
				
				plt.savefig(f"{directory}mean={mean}_bins={bins}.pdf")
				
			plt.clf()
