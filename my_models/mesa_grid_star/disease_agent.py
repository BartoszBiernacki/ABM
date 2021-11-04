from mesa import Agent
import random

from my_math_utils import *


class OrdinaryPearsonAgent(Agent):
    def __init__(self, unique_id, model, household_id, shopping_days):
        super().__init__(unique_id, model)
        self.household_id = household_id
        self.shopping_days = shopping_days
        self.on_shopping = False
        self.did_shopping_today = False
        self.became_infected_today = False
        self.probability_of_staying_alive_after_one_day_of_illness = \
            (1 - model.mortality)**(1/model.avg_illness_period)

        # initial infection with given probability
        if random.uniform(0, 1) <= model.initial_infection_probability:
            self.state = "incubation"
            self.incubation_period = get_rand_int_from_exp_distribution(mean=model.avg_incubation_period)
            self.prodromal_period = 0
            self.illness_period = 0
        else:
            self.state = "susceptible"
            self.incubation_period = 0
            self.prodromal_period = 0
            self.illness_period = 0

    # Helpful agent functions =========================================================================================
    def set_all_disease_periods_to_zero(self):
        self.incubation_period = 0
        self.prodromal_period = 0
        self.illness_period = 0

    def go_into_incubation_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "incubation"
        self.incubation_period = get_rand_int_from_exp_distribution(mean=self.model.avg_incubation_period)

    def go_into_prodromal_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "prodromal"
        self.prodromal_period = get_rand_int_from_exp_distribution(mean=self.model.avg_prodromal_period)

    def go_into_illness_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "illness"
        self.illness_period = get_rand_int_from_exp_distribution(mean=self.model.avg_illness_period)

    def go_into_susceptible_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "susceptible"

    def go_into_dead_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "dead"

    def try_to_die(self):
        if random.uniform(0, 1) > self.probability_of_staying_alive_after_one_day_of_illness:
            self.go_into_dead_phase()

    # SHOPPING CUSTOMER PART ===========================================================================================
    def try_to_infect_cashier(self):
        if self.state == "prodromal":
            cashier = self.model.cashiers_grouped_by_neighbourhood[self.pos]
            if cashier.state == "susceptible":
                if random.uniform(0, 1) <= self.model.beta:
                    cashier.go_into_incubation_phase()

    def try_to_get_infected_by_cashier(self):
        if self.state == "susceptible":
            cashier = self.model.cashiers_grouped_by_neighbourhood[self.pos]
            if cashier.state == "prodromal" or cashier.state == 'illness':
                if random.uniform(0, 1) <= self.model.beta:
                    self.go_into_incubation_phase()
                    self.became_infected_today = True

    def do_shopping(self):
        self.did_shopping_today = False
        self.became_infected_today = False
        if self.on_shopping:
            self.try_to_infect_cashier()
            self.try_to_get_infected_by_cashier()
            self.on_shopping = False
            self.did_shopping_today = True

    # CUSTOMER STEP ====================================================================================================
    def step(self):
        if self.state == "incubation":
            self.incubation_period -= 1
            if self.incubation_period <= 0:
                self.go_into_prodromal_phase()

        elif self.state == "prodromal":
            self.prodromal_period -= 1
            if self.prodromal_period <= 0:
                self.go_into_illness_phase()

        elif self.state == "illness":
            self.illness_period -= 1
            self.try_to_die()
            if self.state != "dead" and self.illness_period <= 0:
                self.go_into_susceptible_phase()

        self.do_shopping()


class CashierAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.probability_of_staying_alive_after_one_day_of_illness = \
            (1 - model.mortality) ** (1 / model.avg_illness_period)

        # initial infection with given probability
        if random.uniform(0, 1) <= model.initial_infection_probability:
            self.state = "incubation"
            self.incubation_period = get_rand_int_from_exp_distribution(mean=model.avg_incubation_period)
            self.prodromal_period = 0
            self.illness_period = 0
        else:
            self.state = "susceptible"
            self.incubation_period = 0
            self.prodromal_period = 0
            self.illness_period = 0

    # Helpful agent functions =========================================================================================
    def set_all_disease_periods_to_zero(self):
        self.incubation_period = 0
        self.prodromal_period = 0
        self.illness_period = 0

    def go_into_incubation_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "incubation"
        self.incubation_period = get_rand_int_from_exp_distribution(mean=self.model.avg_incubation_period)

    def go_into_prodromal_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "prodromal"
        self.prodromal_period = get_rand_int_from_exp_distribution(mean=self.model.avg_prodromal_period)

    def go_into_illness_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "illness"
        self.illness_period = get_rand_int_from_exp_distribution(mean=self.model.avg_illness_period)

    def go_into_susceptible_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "susceptible"

    def go_into_dead_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = "dead"

    def try_to_die(self):
        if random.uniform(0, 1) > self.probability_of_staying_alive_after_one_day_of_illness:
            self.go_into_dead_phase()

    def replace_cashier_if_needed(self):
        if self.state == "illness" or self.state == 'dead':
            # print("Cashier replaced")
            self.go_into_susceptible_phase()

    # Cashier STEP ====================================================================================================
    def step(self):
        if self.state == "incubation":
            self.incubation_period -= 1
            if self.incubation_period <= 0:
                self.go_into_prodromal_phase()

        elif self.state == "prodromal":
            self.prodromal_period -= 1
            if self.prodromal_period <= 0:
                self.go_into_illness_phase()

        elif self.state == "illness":
            self.illness_period -= 1
            self.try_to_die()
            if self.state != "dead" and self.illness_period <= 0:
                self.go_into_susceptible_phase()

        self.replace_cashier_if_needed()


