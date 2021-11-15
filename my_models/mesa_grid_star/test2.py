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

a = [1000, 1000, 300, 400, 500]
b = [0.05, 0.03, 0.06, 0.12, 0.11]
c = [66, 65, 30, 40, 11]

df = pd.DataFrame(np.array([a, b, c]).T, columns=['N', 'beta', 'tau'])

Ns = df['N'].unique()

popt = [65.4, 21.5]

for N in Ns:
	filt = df['N'] == N

	ax = df[filt].plot(x='beta', y='tau', style='.-')
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.set_xlabel(r' $y =  Ae^{-t/\tau}$', fontsize=12)
	ax.set_ylabel(r'$\tau$, days', fontsize=12)
	ax.legend(labels=[r' $y =  Ae^{{-t/\tau}}$' '\n' r'$A={:.1f}$' '\n' r'$\tau = {:.1f}$'.format(popt[0], popt[1])])
	plt.tight_layout()
	plt.show()






