from disease_model import DiseaseModel
from mesa.visualization.modules import CanvasGrid  # to show our grid type implemented in Model
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from disease_agent import OrdinaryPearsonAgent, CashierAgent


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if type(agent) is OrdinaryPearsonAgent:
        # print(f"Agent {agent.unique_id} is {agent.state}")
        if agent.state == "susceptible":
            portrayal["Layer"] = 2
            portrayal["Color"] = "Blue"
            portrayal["r"] = 0.2
        else:
            portrayal["Color"] = "Red"
            portrayal["Layer"] = 0
    if type(agent) is CashierAgent:
        portrayal["Layer"] = 3
        portrayal["Color"] = "Green"
        portrayal["r"] = 0.1
    return portrayal


grid = CanvasGrid(agent_portrayal, grid_width=4, grid_height=4,
                  canvas_width=200, canvas_height=200)


number_of_households_slider = UserSettableParameter('slider', "Number of households",
                                                    value=50, min_value=1, max_value=1000, step=1)
avg_num_of_customers_in_household_slider = UserSettableParameter('slider', "Avg household size",
                                                                 value=3, min_value=1, max_value=10, step=1)
beta_slider = UserSettableParameter('slider', "Beta value", value=0.5, min_value=0.01, max_value=1, step=0.01)

avg_incubation_period_slider = UserSettableParameter('slider', "avg_incubation_period (days)", value=5, min_value=1,
                                                     max_value=10, step=1)
avg_prodromal_period_slider = UserSettableParameter('slider', "avg_prodromal_period (days)", value=3, min_value=1,
                                                    max_value=8, step=1)
avg_illness_period_slider = UserSettableParameter('slider', "avg_illness_period (days)", value=15, min_value=10,
                                                  max_value=20, step=1)
mortality_slider = UserSettableParameter('slider', "Mortality", value=0.1, min_value=0.01, max_value=1, step=0.01)
initial_infection_probability_slider = UserSettableParameter('slider', "initial_infection_probability", value=0.6,
                                                             min_value=0.01, max_value=1, step=0.01)

total_population_situation_graph = ChartModule(
    series=[{"Label": "Incubation people", "Color": "Green"},
            {"Label": "Prodromal people", "Color": "Yellow"},
            {"Label": "Illness people", "Color": "Red"}],
    data_collector_name='datacollector')

total_customers_situation_graph = ChartModule(
    series=[{"Label": "Susceptible customers", "Color": "Green"},
            {"Label": "Incubation customers", "Color": "Yellow"},
            {"Label": "Prodromal customers", "Color": "Red"}],
    data_collector_name='datacollector')

cashier_influence_graph = ChartModule(
    series=[{"Label": "Infected by cashier", "Color": "Red"}],
    data_collector_name='datacollector')


server = ModularServer(model_cls=DiseaseModel,
                       visualization_elements=[grid, total_population_situation_graph,
                                               total_customers_situation_graph, cashier_influence_graph],
                       name="Disease spread model", model_params=
                       {"width": 4, "height": 4,
                        "num_of_households": number_of_households_slider,
                        "avg_num_of_customers_in_household": avg_num_of_customers_in_household_slider,
                        "beta": beta_slider,
                        "avg_incubation_period": avg_incubation_period_slider,
                        "avg_prodromal_period": avg_prodromal_period_slider,
                        "avg_illness_period": avg_illness_period_slider,
                        "mortality" : mortality_slider,
                        "initial_infection_probability": initial_infection_probability_slider,
                        })
