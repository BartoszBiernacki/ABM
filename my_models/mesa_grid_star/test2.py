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
from text_processing import *
from numba import prange
from data_visualisation import *

from pathlib import Path
import pathlib


def plot_1D_death_toll_max_prediction_x_visibility_series_betas(directory, show=True, save=False):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_standard_by_mortality_beta_visibility(directory=directory)
    num_mortalities, num_betas, num_visibilities = grouped_fnames.shape
    
    num_of_lines = num_betas
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
    for plot_id in range(num_mortalities):
        
        mortality = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['mortality']
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        norm_info = ''
        
        main_title = "Death toll prediction by infected toll and mortality" + norm_info + '\n' \
                                                   f"mortality={float(mortality) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('visibility')
        ax.set_ylabel(r'(Infected toll / all people) $\cdot$ visibility $\cdot$ mortality (in percent)')
        
        infected = np.empty((num_betas, num_visibilities))
        betas = {}
        for beta_id in range(num_betas):
            fname = grouped_fnames[plot_id][beta_id][0]
            variable_params = variable_params_from_fname(fname=fname)
            betas[beta_id] = variable_params[r'$\beta$']
            
            visibilities = {}
            for visibility_id in range(num_visibilities):
                fname = grouped_fnames[plot_id][beta_id][visibility_id]
                variable_params = variable_params_from_fname(fname=fname)
                visibilities[visibility_id] = variable_params['visibility']
                
                df = pd.read_csv(fname)
                infected[beta_id][visibility_id] = np.max(df['Dead people']) + np.max(df['Recovery people'])
                infected[beta_id][visibility_id] /= df['Susceptible people'][0]
                infected[beta_id][visibility_id] *= 100
                infected[beta_id][visibility_id] *=\
                    float(variable_params['visibility']) * float(variable_params['mortality'])
            
            label = r'$\beta$='f"{betas[beta_id]}"
            ax.plot(visibilities.values(), infected[beta_id], label=label, color=colors[beta_id],
                    marker='o', markersize=3)
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll prediction by infected toll and mortality'
            save_dir = directory.replace('raw data/', 'plots/')
            save_dir += plot_type + '/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            name = ax.get_title()
            name = name.replace('\n', ' ')
            name = name.replace('    ', ' ')
            name = name.replace(r'$\beta$', 'beta')
            if name[-1:] == ' ':
                name = name[:-1]
            plt.savefig(save_dir + name + '.pdf')
        
        if show:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    
    directory_1D = 'results/Runs=18___Grid_size=(20, 20)___N=1000___Customers_in_household=3___' \
                   'Infected_cashiers_at_start=400___Infect_housemates_boolean=0/' \
                   'raw data/'
    
    directory_1D_v2 = 'results/' \
                      'Runs=17___Grid_size=(20, 20)___N=1000___Customers_in_household=3___' \
                      'Infected_cashiers_at_start=400___Infect_housemates_boolean=0/' \
                      'raw data/'
    
    with cProfile.Profile() as pr:
        x = 1
        plot_1D_death_toll_max_prediction_x_visibility_series_betas(directory=directory_1D, show=False, save=True)
        # plot_1D_infected_max_x_visibility_series_betas(directory=directory_1D, show=False, save=True)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(5)

