import random

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from numba import jit
import time

from my_math_utils import *

from disease_agent import OrdinaryPearsonAgent, CashierAgent
from my_model_utils import calculate_ordinary_pearson_number_incubation
from my_model_utils import calculate_ordinary_pearson_number_prodromal
from my_model_utils import calculate_ordinary_pearson_number_illness
from my_model_utils import calculate_ordinary_pearson_number_dead
from my_model_utils import calculate_ordinary_pearson_number_susceptible
from my_model_utils import calculate_susceptible_customers
from my_model_utils import calculate_incubation_customers
from my_model_utils import calculate_prodromal_customers
from my_model_utils import calculate_incubation_cashiers
from my_model_utils import calculate_prodromal_cashiers
from my_model_utils import calculate_susceptible_cashiers
from my_model_utils import calculate_execution_time
from my_model_utils import get_number_of_total_ordinary_pearson_agents

from my_model_utils import calculate_infected_customers_by_cashier_today


def set_which_household_member_do_shopping(household_members):
    available_members = [agent for agent in household_members if agent.state == "susceptible" or agent.state ==
                         "incubation" or agent.state == "prodromal"]
    if len(available_members) > 0:
        random.choice(available_members).on_shopping = True


class DiseaseModel(Model):
    def __init__(self,  width, height, num_of_households, avg_num_of_customers_in_household,
                 beta, avg_incubation_period, avg_prodromal_period, avg_illness_period, mortality,
                 initial_infection_probability):
        self.running = True  # required for BatchRunner
        self.num_of_households = num_of_households
        self.avg_num_of_customers_in_household = avg_num_of_customers_in_household
        self.beta = beta
        self.avg_incubation_period = avg_incubation_period
        self.avg_prodromal_period = avg_prodromal_period
        self.avg_illness_period = avg_illness_period
        self.mortality = mortality
        self.initial_infection_probability = initial_infection_probability

        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.schedule = RandomActivation(self)
        self.current_id = 0

        self.neighbourhoods = {}
        self.cashiers_grouped_by_neighbourhood = {}
        self.households_grouped_by_neighbourhood = {}
        self.customers_grouped_by_households = {}
        self.current_day = ""

        self.execution_time = 0.
        self.total_num_of_ordinary_agents = 0

        household_id = 0
        available_shopping_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                list_of_households_identifiers = []
                for _ in range(self.num_of_households):
                    list_of_household_members = []
                    shopping_days = random.sample(available_shopping_days, 2)
                    for household_member in range(get_rand_int_from_triangular_distribution(left=1,
                                                                                            mode=self.avg_num_of_customers_in_household,
                                                                                            right=avg_num_of_customers_in_household*2)):
                        a = OrdinaryPearsonAgent(unique_id=self.next_id(), model=self, household_id=household_id,
                                                 shopping_days=shopping_days)
                        self.schedule.add(a)
                        self.total_num_of_ordinary_agents += 1
                        list_of_household_members.append(a)
                        self.grid.place_agent(agent=a, pos=(x, y))
                    self.customers_grouped_by_households[household_id] = np.array(list_of_household_members,
                                                                                  dtype=OrdinaryPearsonAgent)
                    list_of_households_identifiers.append(household_id)
                    household_id += 1

                a = CashierAgent(unique_id=self.next_id(), model=self)
                self.schedule.add(a)
                self.grid.place_agent(agent=a, pos=(x, y))
                self.cashiers_grouped_by_neighbourhood[(x, y)] = a
                self.households_grouped_by_neighbourhood[(x, y)] = np.array(list_of_households_identifiers)

        self.datacollector = DataCollector(
            model_reporters={"Incubation people": calculate_ordinary_pearson_number_incubation,
                             "Prodromal people": calculate_ordinary_pearson_number_prodromal,
                             "Illness people": calculate_ordinary_pearson_number_illness,
                             "Dead people": calculate_ordinary_pearson_number_dead,
                             "Susceptible people": calculate_ordinary_pearson_number_susceptible,
                             "Susceptible customers":  calculate_susceptible_customers,
                             "Incubation customers":  calculate_incubation_customers,
                             "Prodromal customers":  calculate_prodromal_customers,
                             "Incubation cashiers": calculate_incubation_cashiers,
                             "Prodromal cashiers": calculate_prodromal_cashiers,
                             "Susceptible cashiers": calculate_susceptible_cashiers,
                             "Infected by cashier": calculate_infected_customers_by_cashier_today,
                             "Execution time": calculate_execution_time,
                             "Number of ordinary agents": get_number_of_total_ordinary_pearson_agents})

    def set_current_day(self):
        current_day_number = self.schedule.time % 7
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
                5: "Saturday", 6: "Sunday"}
        self.current_day = days[current_day_number]

    def decide_who_does_shopping(self):
        for household_id in range(np.max(list(self.households_grouped_by_neighbourhood.values()))):
            if self.current_day in self.customers_grouped_by_households[household_id][0].shopping_days:
                household_members = self.customers_grouped_by_households[household_id]
                set_which_household_member_do_shopping(household_members=household_members)

    def step(self):
        start_time = time.time()

        self.set_current_day()
        self.decide_who_does_shopping()
        self.schedule.step()
        self.datacollector.collect(model=self)

        end_time = time.time()
        self.execution_time += (end_time - start_time)

