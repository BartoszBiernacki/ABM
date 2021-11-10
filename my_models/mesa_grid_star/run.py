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
from my_math_utils import group_tuples_by_start


def save_average_data_from_batch_runner_to_file(data_collector_model_results, num_of_variable_model_params):
    list_of_tuples = list(data_collector_model_results.keys())
    tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples, start_length=num_of_variable_model_params)

    for key in tuples_grouped.keys():   # key is tuple by which other tuples were grouped. For example key=(5, 2, 2)
        lis = []
        for item in tuples_grouped[key]:    # items are full tuples. For example item=(5, 2, 2, ..., 0)
            lis.append(data_collector_model[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
        array_with_all_iterations_results_for_specific_parameters = np.array(lis)

        average_array = np.mean(array_with_all_iterations_results_for_specific_parameters, axis=0)
        df = pd.DataFrame(data=average_array)
        df.columns = data_collector_model[list_of_tuples[0]].columns
        df["Incubation people"].plot(style='.-')
        (df["Susceptible customers"] + df["Incubation customers"] + df["Prodromal customers"]).plot(style='.-')
        # print(df.to_markdown())
    plt.show()


# BATCH RUNNER SETTINGS----------------------------------------------------------------------------------------------
fixed_params = {"num_of_customers_in_household": 1,
                "num_of_cashiers_in_neighbourhood": 1,
                "avg_incubation_period": 5,
                "avg_prodromal_period": 4,
                "avg_illness_period": 15,
                "mortality": 0.0,
                "initial_infection_probability": 0.7,
                "start_with_infected_cashiers_only": True,
                "random_activation": False,
                "extra_shopping_boolean": True}

variable_params = {"num_of_households_in_neighbourhood": range(1000, 1001, 1),
                   "beta": [0.05],
                   "width": [1],
                   "height": [1]}
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    visualisation = False
    multi = True
    profiling = True
    iterations = 3000

    max_steps = 60
    model_reporters = {"Infected by cashier": calculate_infected_customers_by_cashier_today,
                       "Extra customers": calculate_extra_customers}

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
                                              max_steps=max_steps,
                                              model_reporters=model_reporters)
                    batch_run.run_all()
                    data_collector_model = batch_run.get_collector_model()
                    save_average_data_from_batch_runner_to_file(data_collector_model_results=data_collector_model,
                                                                num_of_variable_model_params=len(variable_params))
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats(10)
            else:
                batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                          nr_processes=os.cpu_count(),
                                          variable_parameters=variable_params,
                                          fixed_parameters=fixed_params,
                                          iterations=iterations,
                                          max_steps=max_steps,
                                          model_reporters=model_reporters)
                batch_run.run_all()
                data_collector_model = batch_run.get_collector_model()
                save_average_data_from_batch_runner_to_file(data_collector_model_results=data_collector_model,
                                                            num_of_variable_model_params=len(variable_params))
        else:
            if profiling:
                with cProfile.Profile() as pr:
                    batch_run = BatchRunner(model_cls=DiseaseModel,
                                            variable_parameters=variable_params,
                                            fixed_parameters=fixed_params,
                                            iterations=iterations,
                                            max_steps=max_steps,
                                            model_reporters=model_reporters)
                    batch_run.run_all()
                    data_collector_model = batch_run.get_collector_model()
                    save_average_data_from_batch_runner_to_file(data_collector_model_results=data_collector_model,
                                                                num_of_variable_model_params=len(variable_params))
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats(10)
            else:
                batch_run = BatchRunner(model_cls=DiseaseModel,
                                        variable_parameters=variable_params,
                                        fixed_parameters=fixed_params,
                                        iterations=iterations,
                                        max_steps=max_steps,
                                        model_reporters=model_reporters)
                batch_run.run_all()
                data_collector_model = batch_run.get_collector_model()
                save_average_data_from_batch_runner_to_file(data_collector_model_results=data_collector_model,
                                                            num_of_variable_model_params=len(variable_params))





# *******************************************************************************************************************
# Plot execution time vs num of agents
# agents = []
# time = []
# for key in data_collector_model.keys():
#     df = data_collector_model[key]
#     num_of_agents = np.array(df["Number of ordinary agents"])[-1]
#     exec_time = np.array(df["Execution time"])[-1]
#     agents.append(num_of_agents)
#     time.append(exec_time)
# plt.scatter(agents, time)
# plt.savefig('size_vs_time_10_iterations.pdf')
# plt.show()
#
# # -------------------------------------------------------------------------------------------------------------------
# # Plot execution time vs num of steps
# for key in data_collector_model.keys():
#     df = data_collector_model[key]
#     df["Execution time"].plot()
# plt.savefig('iterations_vs_time_10k_steps_grid_8_by_8_num_of_households_10_and_11.pdf')
# plt.show()


## *******************************************************************************************************************
# # Save batch runner data to files
# for key in data_collector_model.keys():
#     print(key)
#     df = data_collector_model[key]
#     # print(df.to_markdown())
#     df.to_csv(f'results/{key}.txt', sep='\t', index=False)
# # -------------------------------------------------------------------------------------------------------------------




