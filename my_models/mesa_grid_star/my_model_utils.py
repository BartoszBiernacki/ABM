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


def calculate_susceptible_cashiers(model):
    susceptible_cashier_number = 0
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            if model.cashiers_grouped_by_neighbourhood[(x, y)].state == "susceptible":
                susceptible_cashier_number += 1
    return susceptible_cashier_number


def calculate_incubation_cashiers(model):
    incubation_cashier_number = 0
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            if model.cashiers_grouped_by_neighbourhood[(x, y)].state == "incubation":
                incubation_cashier_number += 1
    return incubation_cashier_number


def calculate_prodromal_cashiers(model):
    prodromal_cashier_number = 0
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            if model.cashiers_grouped_by_neighbourhood[(x, y)].state == "prodromal":
                prodromal_cashier_number += 1
    return prodromal_cashier_number


def calculate_execution_time(model):
    return model.execution_time


def get_number_of_total_ordinary_pearson_agents(model):
    return model.total_num_of_ordinary_agents


def get_initial_agent_state(probability_of_initial_infection):
    val = np.random.rand()
    if val >= probability_of_initial_infection:
        return "susceptible"
    else:
        return "incubation"
