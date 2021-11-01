import random

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

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
from my_model_utils import calculate_infected_customers_by_cashier_today

from my_model_utils import get_current_day


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

        for household_id in range(self.num_of_households):
            available_shopping_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            shopping_days = random.sample(available_shopping_days, 2)
            for _ in range(get_rand_int_from_triangular_distribution(left=1,
                                                                     mode=self.avg_num_of_customers_in_household,
                                                                     right=self.avg_num_of_customers_in_household*2)):
                a = OrdinaryPearsonAgent(unique_id=self.next_id(), model=self, household_id=household_id,
                                         shopping_days=shopping_days)
                self.schedule.add(a)
                try:
                    start_cell = (1, 1)
                    self.grid.place_agent(agent=a, pos=start_cell)
                except:
                    print("PLACING AGENT PROBLEM")

        b = CashierAgent(unique_id=self.next_id(), model=self)
        self.schedule.add(b)
        start_cell = (1, 1)
        self.grid.place_agent(agent=b, pos=start_cell)

        self.datacollector = DataCollector(
            model_reporters={"Incubation people": calculate_ordinary_pearson_number_incubation,
                             "Prodromal people": calculate_ordinary_pearson_number_prodromal,
                             "Illness people": calculate_ordinary_pearson_number_illness,
                             "Dead people": calculate_ordinary_pearson_number_dead,
                             "Susceptible people": calculate_ordinary_pearson_number_susceptible,
                             "Susceptible customers":  calculate_susceptible_customers,
                             "Incubation customers":  calculate_incubation_customers,
                             "Prodromal customers":  calculate_prodromal_customers,
                             "Infected by cashier": calculate_infected_customers_by_cashier_today})


    def decide_who_does_shopping(self):
        for household_id in range(self.num_of_households):
            household_members = [agent for agent in self.schedule.agents if type(agent) is
                                 OrdinaryPearsonAgent and agent.household_id == household_id]
            if get_current_day(model=self) in household_members[0].shopping_days:
                set_which_household_member_do_shopping(household_members=household_members)

    def step(self):
        self.decide_who_does_shopping()
        self.schedule.step()
        self.datacollector.collect(model=self)
