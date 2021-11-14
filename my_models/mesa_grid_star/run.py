import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from mesa.batchrunner import BatchRunner, BatchRunnerMP
from disease_model import DiseaseModel
from my_model_utils import calculate_infected_customers_by_cashier_today
from my_model_utils import calculate_extra_customers
from disease_server import server
from my_math_utils import group_tuples_by_start, fit_exp_to_peaks


def run_simulation(variable_params, fixed_params, visualisation, multi, profiling, iterations, max_steps):
    if visualisation:
        server.port = 8521  # default
        server.launch()
    else:
        if multi:
            if profiling:
                with cProfile.Profile() as pr:
                    batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                              nr_processes=os.cpu_count(),
                                              variable_parameters=variable_params,
                                              fixed_parameters=fixed_params,
                                              iterations=iterations,
                                              max_steps=max_steps)
                    batch_run.run_all()
                    data_collector_model = batch_run.get_collector_model()
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats(10)
            else:
                batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                          nr_processes=os.cpu_count(),
                                          variable_parameters=variable_params,
                                          fixed_parameters=fixed_params,
                                          iterations=iterations,
                                          max_steps=max_steps)
                batch_run.run_all()
                data_collector_model = batch_run.get_collector_model()
        else:
            if profiling:
                with cProfile.Profile() as pr:
                    batch_run = BatchRunner(model_cls=DiseaseModel,
                                            variable_parameters=variable_params,
                                            fixed_parameters=fixed_params,
                                            iterations=iterations,
                                            max_steps=max_steps)
                    batch_run.run_all()
                    data_collector_model = batch_run.get_collector_model()
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats(10)
            else:
                batch_run = BatchRunner(model_cls=DiseaseModel,
                                        variable_parameters=variable_params,
                                        fixed_parameters=fixed_params,
                                        iterations=iterations,
                                        max_steps=max_steps)
                batch_run.run_all()
                data_collector_model = batch_run.get_collector_model()

        return get_avg_results(data_collector_model_results=data_collector_model, variable_params=variable_params)


def get_avg_results(data_collector_model_results, variable_params):
    # returns dict in which keys are tuples of variable_params and values are dataframes averaged over all iterations
    num_of_variable_model_params = len(variable_params)
    list_of_tuples = list(data_collector_model_results.keys())
    tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples, start_length=num_of_variable_model_params)

    result = {}
    for key in tuples_grouped.keys():  # key is a tuple by which other tuples were grouped. For example key=(5, 2, 2)
        lis = []
        for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
            lis.append(data_collector_model_results[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
        array_with_all_iterations_results_for_specific_parameters = np.array(lis)

        average_array = np.mean(array_with_all_iterations_results_for_specific_parameters, axis=0)
        df = pd.DataFrame(data=average_array)
        df.columns = data_collector_model_results[list_of_tuples[0]].columns
        result[key] = df

    return result


def save_average_data_from_batch_runner_to_file(data_collector_model_results, variable_params):
    results = get_avg_results(data_collector_model_results=data_collector_model_results,
                              variable_params=variable_params)

    print(results)

    # tau = fit_exp_to_peaks(y_data=df["Incubation people"], plot=True)
    #
    #
    #
    #
    # print("Max number of susceptible people = ", np.max(df["Incubation people"]), " while it should be 55.93")
    # # (df["Susceptible customers"] + df["Incubation customers"] + df["Prodromal customers"]).plot(style='.-')
    # # print(df.to_markdown())
    # plt.show()


def find_tau_for_given_Ns_and_betas(Ns, betas, iterations, max_steps, plot_results=True):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
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

    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta": betas}
    # ---------------------------------------------------------------------------------------------------------------------

    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=False,
                             multi=True,
                             profiling=False,
                             iterations=iterations,
                             max_steps=max_steps)

    for key in results.keys():
        avg_df = results[key]
        print(avg_df)


    print(results)


if __name__ == '__main__':
    find_tau_for_given_Ns_and_betas(Ns=[1000], betas=[0.05, 0.08],
                                    iterations=15, max_steps=150)

## *******************************************************************************************************************
# # Save batch runner data to files
# for key in data_collector_model.keys():
#     print(key)
#     df = data_collector_model[key]
#     # print(df.to_markdown())
#     df.to_csv(f'results/{key}.txt', sep='\t', index=False)
# # -------------------------------------------------------------------------------------------------------------------
