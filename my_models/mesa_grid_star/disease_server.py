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
        if agent.state == 0:
            portrayal["Layer"] = 2
            portrayal["Color"] = "Blue"
            portrayal["r"] = 0.2
        else:
            portrayal["Color"] = "Red"
            portrayal["Layer"] = 0
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

number_of_households_slider = UserSettableParameter('slider', "Number of households",
                                                    value=10, min_value=1, max_value=1000, step=5)
avg_num_of_customers_in_household_slider = UserSettableParameter('slider', "Avg household size",
                                                                 value=1, min_value=1, max_value=10, step=1)
beta_slider = UserSettableParameter('slider', "Beta value", value=0.05, min_value=0.0, max_value=1, step=0.01)

avg_incubation_period_slider = UserSettableParameter('slider', "avg_incubation_period (days)", value=5, min_value=1,
                                                     max_value=10, step=1)
avg_prodromal_period_slider = UserSettableParameter('slider', "avg_prodromal_period (days)", value=3, min_value=1,
                                                    max_value=8, step=1)
avg_illness_period_slider = UserSettableParameter('slider', "avg_illness_period (days)", value=15, min_value=10,
                                                  max_value=20, step=1)
mortality_slider = UserSettableParameter('slider', "Mortality", value=0.1, min_value=0.0, max_value=1, step=0.01)
initial_infection_probability_slider = UserSettableParameter('slider', "initial_infection_probability", value=0.6,
                                                             min_value=0.01, max_value=1, step=0.01)

start_with_infected_cashiers_only_switch = UserSettableParameter('checkbox', 'start with infected cashiers only',
                                                                 value=True)

random_activation_switch = UserSettableParameter('checkbox', 'active agents in random order', value=True)
extra_shopping_boolean_switch = UserSettableParameter('checkbox', 'allow to extra shopping', value=True)


exposed_population_graph = ChartModule(
    series=[{"Label": "Incubation people", "Color": "Yellow"}],
    data_collector_name='datacollector')

extra_customers_graph = ChartModule(
    series=[{"Label": "Extra customers", "Color": "Black"}],
    data_collector_name='datacollector')


total_population_situation_graph = ChartModule(
    series=[{"Label": "Susceptible people", "Color": "Green"},
            {"Label": "Incubation people", "Color": "Yellow"},
            {"Label": "Prodromal people", "Color": "Orange"},
            {"Label": "Illness people", "Color": "Red"},
            {"Label": "Dead people", "Color": "Black"},
            {"Label": "Recovery people", "Color": "Blue"}],
    data_collector_name='datacollector')

total_customers_situation_graph = ChartModule(
    series=[{"Label": "Susceptible customers", "Color": "Green"},
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


server = ModularServer(model_cls=DiseaseModel,
                       visualization_elements=[grid,
                                               exposed_population_graph,
                                               extra_customers_graph,
                                               total_population_situation_graph,
                                               total_customers_situation_graph,
                                               total_cashiers_situation_graph,
                                               cashier_influence_graph],
                       name="Disease spread model", model_params=
                       {"height": height_slider,
                        "width": width_slider,
                        "num_of_households_in_neighbourhood": number_of_households_slider,
                        "num_of_customers_in_household": avg_num_of_customers_in_household_slider,
                        "num_of_cashiers_in_neighbourhood": 1,
                        "beta": beta_slider,
                        "avg_incubation_period": avg_incubation_period_slider,
                        "avg_prodromal_period": avg_prodromal_period_slider,
                        "avg_illness_period": avg_illness_period_slider,
                        "mortality": mortality_slider,
                        "initial_infection_probability": initial_infection_probability_slider,
                        "start_with_infected_cashiers_only": start_with_infected_cashiers_only_switch,
                        "random_activation": random_activation_switch,
                        "extra_shopping_boolean": extra_shopping_boolean_switch
                        })
