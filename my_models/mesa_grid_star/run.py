import cProfile
import itertools
import pstats

import numpy as np

from my_math_utils import calc_exec_time
from my_model_data_analysis import run_and_save_simulations
from disease_server import run_simulation_in_browser
from data_visualisation import reproduce_plot_by_website


grid_size = (20, 20)
N = 1000
customers_in_household = 3
max_steps = 500
iterations = 18
beta_sweep = (0.02, 0.07, 11)
mortality_sweep = (0.5/100, 1.5/100, 3)
visibility_sweep = (0.2, 1., 17)
run = False

if __name__ == '__main__':
    calc_exec_time(grid_size=grid_size, N=N, max_steps=max_steps, iterations=iterations,
                   household_size=customers_in_household,
                   betas=beta_sweep[2], mortalities=mortality_sweep[2], visibilities=visibility_sweep[2])
    
    directory_700 = 'results/Runs=1000___Grid_size=(1, 1)___N=700___Num_of_infected_cashiers_at_start=1/'
    directory_1000 = 'results/Runs=1000___Grid_size=(1, 1)___N=1000___Num_of_infected_cashiers_at_start=1/'
    
    directory_1D = 'results/Runs=51___Grid_size=(20, 20)___N=1000___Num_of_customers_in_household=3___' \
                   'Num_of_infected_cashiers_at_start=400/'
    directory_2D = 'results/Runs=51___Grid_size=(20, 20)___N=1000___Num_of_customers_in_household=3___' \
                   'Num_of_infected_cashiers_at_start=400/'
    
    fname_20x20 = directory_2D + 'Id=0052___Beta_mortality_pair=(0.05, 0.011).csv'
    fname_1x1 = 'results/Runs=2220___Grid_size=(1, 1)___N=1000___Num_of_infected_cashiers_at_start=1/' \
                'Id=0000___Beta_mortality_pair=(0.05, 0.01).csv'

    fname = directory_1D + 'Id=0004___Beta_mortality_pair=(0.02, 0.011).csv'
    
    beta_mortality = [[(beta, mortality) for beta in np.linspace(*beta_sweep)]
                      for mortality in np.linspace(*mortality_sweep)]
    # -----------------------------------------------------------------------------------------------------------------

    # plot_1D_death_toll(directory=directory_1D)
    # plot_2D_death_toll(directory=directory_2D)

    # plot_1D_death_toll_const_mortality(directory=directory_1D, const_mortality=1.1/100, normalized=True)
    # plot_1D_death_toll_const_mortality(directory=directory_1D, const_mortality=1.1/100, normalized=False)

    # plot_1D_death_toll_const_beta(directory=directory_1D, const_beta=0.05, normalized=True)
    # plot_1D_death_toll_const_beta(directory=directory_1D, const_beta=0.05, normalized=False)

    # plot_1D_pandemic_time(directory=directory_1D)
    # plot_2D_pandemic_time(directory=directory_2D)

    # plot_tau_exp_fitting(fname=fname_20x20, days=300)
    # plot_1D_tau(directory=directory_2D)
    # plot_1D_tau(directory='results/Runs=500___Grid_size=(1, '
    #                       '1)___N=1000___Num_of_customers_in_household=3___Num_of_infected_cashiers_at_start=1/')
    # plot_tau_700_and_1000(directory_700=directory_700, directory_1000=directory_1000)
    
    # plot_fraction_of_susceptible(fname=fname)

    if run:
        with cProfile.Profile() as pr:
            run_and_save_simulations(
                fixed_params={
                    "grid_size": grid_size,
                    "N": N,
                    "customers_in_household": customers_in_household,
    
                    "avg_incubation_period": 5,
                    "incubation_period_bins": 3,
                    "avg_prodromal_period": 3,
                    "prodromal_period_bins": 3,
                    "avg_illness_period": 15,
                    "illness_period_bins": 1,
    
                    "die_at_once": False,
                    "infected_cashiers_at_start": grid_size[0]*grid_size[1],
                    "infect_housemates_boolean": False,
                    "extra_shopping_boolean": True
                },
    
                variable_params={
                    "beta_mortality_pair": list(itertools.chain.from_iterable(beta_mortality)),
                    "visibility": list(np.linspace(*visibility_sweep)),
                },
    
                save_dir='results/',
                iterations=iterations,
                max_steps=max_steps,
    
                base_params=['grid_size',
                             'N',
                             'infected_cashiers_at_start',
                             'customers_in_household',
                             'infect_housemates_boolean']
            )
            
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(10)

       

    # run_simulation_in_browser()

    # reproduce_plot_by_website()
