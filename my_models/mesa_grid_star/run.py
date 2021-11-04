import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt


from mesa.batchrunner import BatchRunner
from disease_model import DiseaseModel
from my_model_utils import calculate_infected_customers_by_cashier_today
from disease_server import server


# RUN VISUALISATION --------------------------------------------------------------------------------------------------
server.port = 8521  # default
server.launch()
# -------------------------------------------------------------------------------------------------------------------


# RUN BATCH RUNNER --------------------------------------------------------------------------------------------------
# fixed_params = {"avg_num_of_customers_in_household": 3,
#                 "avg_incubation_period": 5, "avg_prodromal_period": 3, "avg_illness_period": 15,
#                 "mortality": 0.01, "initial_infection_probability": 0.7, "beta": 0.3}
# variable_params = {"num_of_households": range(10, 12, 1),
#                    "width": [8],
#                    "height": [8]}
#
# batch_run = BatchRunner(DiseaseModel, variable_params, fixed_params, iterations=1, max_steps=10000,
#                         model_reporters={"Infected by cashier": calculate_infected_customers_by_cashier_today})
#
# with cProfile.Profile() as pr:
#     batch_run.run_all()
#     data_collector_model = batch_run.get_collector_model()
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats(10)
## *******************************************************************************************************************
# # Plot execution time vs num of agents
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




