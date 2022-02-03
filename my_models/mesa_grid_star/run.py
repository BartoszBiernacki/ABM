import cProfile
import itertools
import pstats
import numpy as np

from my_math_utils import calc_exec_time
from my_model_data_analysis import run_and_save_simulations
from data_visualisation import plot_1D_death_toll_dynamic_real_and_beta_sweep, \
    plot_1D_recovered_dynamic_real_and_beta_sweep, plot_stochastic_1D_death_toll_dynamic, \
    plot_auto_fit_result, show_real_death_toll_voivodeship_shifted_by_hand
from data_visualisation import plot_all_possible_plots
from disease_server import run_simulation_in_browser
from data_visualisation import reproduce_plot_by_website
from my_models.mesa_grid_star.text_processing import all_fnames_from_dir
from real_data import RealData
from avg_results import remove_tmp_results, get_single_results
from config import Config

customers_in_household = 3
real_data_obj = RealData(customers_in_household=customers_in_household)
real_general_data = real_data_obj.get_real_general_data()
real_death_toll = real_data_obj.get_real_death_toll()
real_infected_toll = real_data_obj.get_real_infected_toll()


voivodeship = Config.voivodeship
N = real_general_data.loc[voivodeship, 'N MODEL']
grid_side_length = real_general_data.loc[voivodeship, 'grid side MODEL']
grid_size = (grid_side_length, grid_side_length)
max_steps = 250

infected_cashiers_at_start = grid_side_length

iterations = 10
beta_sweep = (0.012, 0.030, 1)
beta_changes = ((1000, 2000), (1., 1.))
mortality_sweep = (2/100, 2./100, 1)
visibility_sweep = (0.65, 1., 1)
run = True
plot_all = False


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
            directory = run_and_save_simulations(
                fixed_params={
                    "grid_size": grid_size,
                    "N": N,
                    "customers_in_household": customers_in_household,
                    "beta_changes": beta_changes,
    
                    "avg_incubation_period": 5,
                    "incubation_period_bins": 3,
                    "avg_prodromal_period": 3,
                    "prodromal_period_bins": 3,
                    "avg_illness_period": 15,
                    "illness_period_bins": 1,
    
                    "die_at_once": False,
                    "infected_cashiers_at_start": infected_cashiers_at_start,
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
        # stats.print_stats(10)
        
        remove_tmp_results()
        if plot_all:
            show_real_death_toll_voivodeship_shifted_by_hand(directory_to_data=directory,
                                                             voivodeship=voivodeship,
                                                             starting_day=10,
                                                             day_in_which_colors_are_set=60,
                                                             last_day=100,
                                                             shift_simulated=True,
                                                             save=True,
                                                             show=True)
            
         
            
      
    


            # plot_all_possible_plots(directory=directory)

    
    # run_simulation_in_browser()

    # reproduce_plot_by_website()
