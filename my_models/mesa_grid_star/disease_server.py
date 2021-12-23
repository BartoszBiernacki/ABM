from disease_model import DiseaseModel
from mesa.visualization.modules import CanvasGrid  # to show our grid type implemented in Model
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from disease_agent import CashierAgent


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if type(agent) is CashierAgent:
        portrayal["Layer"] = 3
        portrayal["r"] = 0.7
        if agent.state == 0:
            portrayal["Color"] = "Green"
        elif agent.state == 1:
            portrayal["Color"] = "Yellow"
        elif agent.state == 2:
            portrayal["Color"] = "Red"
        else:
            portrayal["Color"] = "Black"

    return portrayal


grid = CanvasGrid(agent_portrayal, grid_width=20, grid_height=20,
                  canvas_width=500, canvas_height=500)


height_slider = UserSettableParameter('slider', "Grid height", value=3, min_value=1, max_value=20, step=1)
width_slider = UserSettableParameter('slider', "Grid width", value=3, min_value=1, max_value=20, step=1)

num_of_infected_cashiers_at_start_slider = UserSettableParameter('slider', "Number of infected cashiers at start",
                                                                 value=3, min_value=1, max_value=400, step=1)

number_of_households_slider = UserSettableParameter('slider', "Number of households",
                                                    value=1000, min_value=1, max_value=1000, step=5)
avg_num_of_customers_in_household_slider = UserSettableParameter('slider', "Avg household size",
                                                                 value=1, min_value=1, max_value=10, step=1)
beta_slider = UserSettableParameter('slider', "Beta value", value=0.05, min_value=0.0, max_value=1, step=0.01)

avg_incubation_period_slider = UserSettableParameter('slider', "avg_incubation_period (days)", value=5, min_value=1,
                                                     max_value=10, step=1)

incubation_period_bins_slider = UserSettableParameter('slider', "Num of possible incubation period values",
                                                      value=1, min_value=1, max_value=9, step=2)

avg_prodromal_period_slider = UserSettableParameter('slider', "avg_prodromal_period (days)", value=3, min_value=1,
                                                    max_value=8, step=1)
prodromal_period_bins_slider = UserSettableParameter('slider', "Num of possible prodromal period values", value=1,
                                                     min_value=1, max_value=9, step=2)

avg_illness_period_slider = UserSettableParameter('slider', "avg_illness_period (days)", value=15, min_value=10,
                                                  max_value=20, step=1)
illness_period_bins_slider = UserSettableParameter('slider', "Num of possible illness period values", value=1,
                                                   min_value=1, max_value=9, step=2)

mortality_slider = UserSettableParameter('slider', "Mortality", value=0.1, min_value=0.0, max_value=1, step=0.01)

visibility_slider = UserSettableParameter('slider', "Visibility", value=0.6, min_value=0.0, max_value=1, step=0.01)

die_at_once_switch = UserSettableParameter('checkbox', 'Die at once', value=False)

extra_shopping_boolean_switch = UserSettableParameter('checkbox', 'allow to extra shopping', value=True)
infect_housemates_boolean_switch = UserSettableParameter('checkbox', 'Enable housemates infections', value=True)


exposed_population_graph = ChartModule(
    series=[{"Label": "Incubation people", "Color": "Yellow"}],
    data_collector_name='datacollector')

extra_customers_graph = ChartModule(
    series=[{"Label": "Extra customers", "Color": "Black"}],
    data_collector_name='datacollector')


total_population_situation_graph = ChartModule(
    series=[{"Label": "Incubation people", "Color": "Yellow"},
            {"Label": "Prodromal people", "Color": "Orange"},
            {"Label": "Illness people", "Color": "Red"},
            {"Label": "Dead people", "Color": "Black"},
            {"Label": "Recovery people", "Color": "Blue"}],
    data_collector_name='datacollector')

total_customers_situation_graph = ChartModule(
    series=[
            {"Label": "Incubation customers", "Color": "Yellow"},
            {"Label": "Prodromal customers", "Color": "Red"},
            {"Label": "Recovery customers", "Color": "Blue"}],
    data_collector_name='datacollector')

total_cashiers_situation_graph = ChartModule(
    series=[{"Label": "Susceptible cashiers", "Color": "Green"},
            {"Label": "Incubation cashiers", "Color": "Yellow"},
            {"Label": "Prodromal cashiers", "Color": "Red"}],
    data_collector_name='datacollector')

cashier_influence_graph = ChartModule(
    series=[{"Label": "Infected by cashier", "Color": "Red"}],
    data_collector_name='datacollector')


def run_simulation_in_browser():
    server = ModularServer(model_cls=DiseaseModel,
                           visualization_elements=[grid,
                                                   exposed_population_graph,
                                                   extra_customers_graph,
                                                   total_population_situation_graph,
                                                   total_customers_situation_graph,
                                                   total_cashiers_situation_graph,
                                                   cashier_influence_graph
                                                   ],
                           name="Disease spread model", model_params=
                           {"width": width_slider,
                            "height": height_slider,
                            "grid_size": (1, 1),  # will be ignored
    
                            "N": number_of_households_slider,
                            "num_of_customers_in_household": avg_num_of_customers_in_household_slider,
    
                            "beta": beta_slider,
                            "mortality": mortality_slider,
                            "beta_mortality_pair": (0.05, 1 / 100),  # will be ignored
                            "visibility": visibility_slider,
    
                            "avg_incubation_period": avg_incubation_period_slider,
                            "incubation_period_bins": incubation_period_bins_slider,
                            "avg_prodromal_period": avg_prodromal_period_slider,
                            "prodromal_period_bins": prodromal_period_bins_slider,
                            "avg_illness_period": avg_illness_period_slider,
                            "illness_period_bins": illness_period_bins_slider,
                            "num_of_infected_cashiers_at_start": num_of_infected_cashiers_at_start_slider,
    
                            "die_at_once": die_at_once_switch,
                            "extra_shopping_boolean": extra_shopping_boolean_switch,
                            "infect_housemates_boolean": infect_housemates_boolean_switch,
                            "max_steps": 1000,
                            })
    
    server.port = 8521  # default
    server.launch()
