from disease_agent import OrdinaryPearsonAgent, CashierAgent
from my_math_utils import *


def calculate_ordinary_pearson_number_susceptible(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.state == 'susceptible'])


def calculate_ordinary_pearson_number_incubation(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.state == 'incubation'])


def calculate_ordinary_pearson_number_prodromal(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.state == 'prodromal'])


def calculate_ordinary_pearson_number_illness(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.state == 'illness'])


def calculate_ordinary_pearson_number_dead(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.state == 'dead'])


def calculate_susceptible_customers(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.did_shopping_today and agent.state ==
                'susceptible'])


def calculate_incubation_customers(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.did_shopping_today and agent.state ==
                'incubation'])


def calculate_prodromal_customers(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.did_shopping_today and agent.state ==
                'prodromal'])


def calculate_infected_customers_by_cashier_today(model):
    ordinary_pearson_agents = [agent for agent in model.schedule.agents if type(agent) is OrdinaryPearsonAgent]
    return len([agent for agent in ordinary_pearson_agents if agent.became_infected_today])


def get_current_day(model):
    days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
            5: "Saturday", 6: "Sunday"}
    current_day_number = model.schedule.time % 7

    return days[current_day_number]


def get_initial_agent_state(probability_of_initial_infection):
    val = np.random.rand()
    if val >= probability_of_initial_infection:
        return "susceptible"
    else:
        return "incubation"
