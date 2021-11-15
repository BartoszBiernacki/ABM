from my_model_data_analysis import run_simulation
from my_model_data_analysis import find_tau_for_given_Ns_and_betas


# Just run simulation (without analysing data except for average model reporters) *************************************
# BATCH RUNNER SETTINGS, FOR just_run_simulation() ONLY!!! ------------------------------------------------------------
fixed_params = {"width": 1,
                "height": 1,
                "num_of_customers_in_household": 1,
                "num_of_cashiers_in_neighbourhood": 1,
                "avg_incubation_period": 5,
                "avg_prodromal_period": 4,
                "avg_illness_period": 15,
                "mortality": 0.0,
                "initial_infection_probability": 0.7,
                "start_with_infected_cashiers_only": True,
                "random_activation": True,
                "extra_shopping_boolean": False}

variable_params = {"num_of_households_in_neighbourhood": [1000],
                   "beta": [0.05]}
iterations = 5000
max_steps = 30
# ---------------------------------------------------------------------------------------------------------------------


def just_run_simulation():
    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=True,
                             multi=False,
                             profiling=False,
                             iterations=iterations,
                             max_steps=max_steps,
                             modified_brMP=False)
    return results
# *********************************************************************************************************************


if __name__ == '__main__':
    # just_run_simulation()     ## runs interactive simulation in browser

    find_tau_for_given_Ns_and_betas(Ns=[1000, 700],
                                    betas=[0.05, 0.06, 0.07],
                                    iterations=50,
                                    max_steps=150,
                                    plot_exp_fittings=True,
                                    plot_tau_vs_beta_for_each_N=True,
                                    modified_brMP=False,
                                    random_activation=True)
