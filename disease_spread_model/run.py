import itertools
import numpy as np

from model.my_math_utils import calc_exec_time
from model.model_runs import RunModel
from data_processing.real_data import RealData
from config import Config


voivodeship = Config.voivodeship
N = RealData.get_real_general_data().loc[voivodeship, 'N MODEL']
grid_side_length = RealData.get_real_general_data().loc[voivodeship, 'grid side MODEL']
grid_size = (grid_side_length, grid_side_length)
infected_cashiers_at_start = grid_side_length

max_steps = 250
iterations = 6
beta_sweep = (0.012, 0.030, 1)
mortality_sweep = (2/100, 2./100, 1)
visibility_sweep = (0.65, 1., 1)
run = True
remove_tmp = False


if __name__ == '__main__':
    calc_exec_time(grid_size=grid_size, N=N, max_steps=max_steps, iterations=iterations,
                   household_size=Config.customers_in_household,
                   betas=beta_sweep[2], mortalities=mortality_sweep[2], visibilities=visibility_sweep[2])
    
    beta_mortality = [[(beta, mortality) for beta in np.linspace(*beta_sweep)]
                      for mortality in np.linspace(*mortality_sweep)]
    # ----------------------------------------------------------------------------------------------------

    if run:
        directory = RunModel.run_and_save_simulations(
            fixed_params={
                "grid_size": grid_size,
                "N": N,
                "customers_in_household": Config.customers_in_household,

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

            iterations=iterations,
            max_steps=max_steps,

            base_params=['grid_size',
                         'N',
                         'infected_cashiers_at_start',
                         'customers_in_household',
                         'infect_housemates_boolean'],
            
            remove_single_results=remove_tmp
        )
        
    # run_simulation_in_browser()
