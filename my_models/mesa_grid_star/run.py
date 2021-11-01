import cProfile
import pstats

from mesa.batchrunner import BatchRunner
from disease_model import DiseaseModel
from my_model_utils import calculate_infected_customers_by_cashier_today


from disease_server import server

# server.port = 8521  # default
# server.launch()


fixed_params = {"width": 5, "height": 5, "avg_num_of_customers_in_household": 3,
                "beta": 0.5, "avg_incubation_period": 5, "avg_prodromal_period": 3, "avg_illness_period": 15,
                "mortality": 0.01, "initial_infection_probability": 0.7}
variable_params = {"num_of_households": range(200, 220, 5)}

batch_run = BatchRunner(DiseaseModel, variable_params, fixed_params, iterations=1, max_steps=200,
                        model_reporters={"Infected by cashier": calculate_infected_customers_by_cashier_today})

with cProfile.Profile() as pr:
    batch_run.run_all()
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(5)

data_collector_model = batch_run.get_collector_model()

# for i, df in enumerate(data_collector_model.values()):
#     print("DUPA START")
#     print(df.to_markdown())
#     df.to_csv(f'file_{i}.txt', sep='\t', index=False)