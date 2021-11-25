from mesa import Agent

from my_math_utils import *

from my_model_utils import get_incubation_period
from my_model_utils import get_prodromal_period
from my_model_utils import get_illness_period


class OrdinaryPearsonAgent(Agent):
    def __init__(self, unique_id, model, number_in_household, household_id, neighbourhood_id):
        super().__init__(unique_id, model)
        self.number_in_household = number_in_household
        self.household_id = household_id
        self.neighbourhood_id = neighbourhood_id
        self.on_shopping = False
        self.on_extra_shopping = False

        self.go_into_susceptible_today = False
        self.go_into_incubation_today = False
        self.go_into_prodromal_today = False
        self.go_into_illness_today = False
        self.go_into_recovery_today = False
        self.go_into_dead_today = False

        self.asSC = 0
        self.asPD = 0
        self.days = []
        self.did_shopping_today = False

        if self.model.start_with_infected_cashiers_only:
            self.state = 0
            self.model.ordinary_pearson_number_susceptible += 1
            self.incubation_period = 0
            self.prodromal_period = 0
            self.illness_period = 0
        else:
            # initial infection with given probability
            if np.random.rand() > model.initial_infection_probability:
                self.state = 0
                self.model.ordinary_pearson_number_susceptible += 1
                self.incubation_period = 0
                self.prodromal_period = 0
                self.illness_period = 0
            else:
                self.state = 1
                self.model.ordinary_pearson_number_incubation += 1
                self.incubation_period = get_incubation_period(model=model)
                self.prodromal_period = 0
                self.illness_period = 0

    # Helpful agent methods =========================================================================================
    def infect_housemates(self):
        housemates = self.model.customers_grouped_by_households[self.household_id].copy()
        housemates.remove(self)
        if housemates:
            [housemate.go_into_incubation_phase() for housemate in housemates if housemate.state == 0]

    def set_all_disease_periods_to_zero(self):
        self.incubation_period = 0
        self.prodromal_period = 0
        self.illness_period = 0

    def go_into_susceptible_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 0
        self.model.ordinary_pearson_number_susceptible += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state

    def go_into_incubation_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 1
        self.model.ordinary_pearson_number_susceptible -= 1     # only susceptible can go into incubation
        self.model.ordinary_pearson_number_incubation += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state
        self.incubation_period = get_incubation_period(model=self.model)

    def go_into_prodromal_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 2
        self.model.ordinary_pearson_number_incubation -= 1
        self.model.ordinary_pearson_number_prodromal += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state
        self.prodromal_period = get_prodromal_period(model=self.model)

        self.try_to_infect_housemates()

    def go_into_illness_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 3
        self.model.ordinary_pearson_number_prodromal -= 1
        self.model.ordinary_pearson_number_illness += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state
        self.illness_period = get_illness_period(model=self.model)

    def go_into_recovery_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = -1
        self.model.ordinary_pearson_number_illness -= 1
        self.model.ordinary_pearson_number_recovery += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state

    def go_into_dead_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 5
        self.model.ordinary_pearson_number_illness -= 1
        self.model.ordinary_pearson_number_dead += 1
        self.model.agents_state_grouped_by_households[self.household_id][self.number_in_household] = self.state

    def try_to_die_after_one_day(self):
        if np.random.rand() > self.model.probability_of_staying_alive_after_one_day_of_illness:
            self.go_into_dead_today = True

    def try_to_die_at_once_or_recover(self):
        if np.random.rand() <= self.model.mortality:
            self.go_into_dead_today = True
        else:
            self.go_into_recovery_today = True

    def is_health_enough_to_do_shopping(self):
        return self.state <= 2

    # SHOPPING CUSTOMER PART ===========================================================================================
    def try_to_infect_cashier(self, neighbourhood_id):
        if self.state == 2:
            # In that state of development each neighbourhood has exactly one cashier so I used a shortcut to access him
            cashier = self.model.cashiers_grouped_by_neighbourhood_id[neighbourhood_id][0]
            if cashier.state == 0:
                if np.random.rand() <= self.model.beta:
                    cashier.go_into_incubation_today = True
                    self.model.cashiers_infected_by_customers_today += 1

    def try_to_get_infected_by_cashier(self, neighbourhood_id):
        if self.state == 0:
            # In that state of development each neighbourhood has exactly one cashier so I used a shortcut to access him
            cashier = self.model.cashiers_grouped_by_neighbourhood_id[neighbourhood_id][0]
            if cashier.state == 2:
                if np.random.rand() <= self.model.beta:
                    self.go_into_incubation_today = True
                    self.model.customers_infected_by_cashier_today += 1

    def do_shopping(self):
        if self.on_shopping:
            self.try_to_infect_cashier(self.neighbourhood_id)
            self.try_to_get_infected_by_cashier(self.neighbourhood_id)
            self.on_shopping = False
            self.did_shopping_today = True
            if self.state == 0:
                self.model.susceptible_customers += 1
            elif self.state == 1:
                self.asSC += 1
                self.days.append(self.model.day)
                self.model.incubation_customers += 1
            elif self.state == 2:
                self.asPD += 1
                self.days.append(self.model.day)
                self.model.prodromal_customers += 1
            elif self.state == -1:
                self.model.recovery_customers += 1

        if self.model.extra_shopping_boolean:
            if self.on_extra_shopping:
                self.model.extra_customers += 1
                neighbouring_neighbourhood_id =\
                    self.model.order_of_shopping_in_neighbouring_neighbourhoods[self.neighbourhood_id][self.number_in_household][self.model.day // 7]

                self.try_to_infect_cashier(neighbourhood_id=neighbouring_neighbourhood_id)
                self.try_to_get_infected_by_cashier(neighbourhood_id=neighbouring_neighbourhood_id)
                self.on_extra_shopping = False

                if self.state == 0:
                    self.model.susceptible_extra_customers += 1
                elif self.state == 1:
                    self.model.incubation_extra_customers += 1
                elif self.state == 2:
                    self.model.prodromal_extra_customers += 1
                elif self.state == -1:
                    self.model.recovery_extra_customers += 1

    def try_to_infect_housemates(self):
        housemates = self.model.customers_grouped_by_households[self.household_id]
        housemates_to_infect = [housemate for housemate in housemates if housemate.state == 0]

        if housemates_to_infect:
            for housemate in housemates_to_infect:
                housemate.go_into_incubation_today = True

    def update_state(self):
        if self.go_into_susceptible_today:
            self.go_into_susceptible_phase()
            self.go_into_susceptible_today = False

        elif self.go_into_incubation_today:
            self.go_into_incubation_phase()
            self.go_into_incubation_today = False

        elif self.go_into_prodromal_today:
            self.go_into_prodromal_phase()
            self.go_into_prodromal_today = False

        elif self.go_into_illness_today:
            self.go_into_illness_phase()
            self.go_into_illness_today = False

        elif self.go_into_recovery_today:
            self.go_into_recovery_phase()
            self.go_into_recovery_today = False

        elif self.go_into_dead_today:
            self.go_into_dead_phase()
            self.go_into_dead_today = False

    # CUSTOMER STEP ====================================================================================================
    def step(self):
        if self.state == 1:
            self.incubation_period -= 1
            if self.incubation_period <= 0:
                self.go_into_prodromal_today = True

        elif self.state == 2:
            self.prodromal_period -= 1
            if self.prodromal_period <= 0:
                self.go_into_illness_today = True

        elif self.state == 3:
            if self.model.die_at_once:
                self.illness_period -= 1
                if self.illness_period <= 0:
                    self.try_to_die_at_once_or_recover()
            else:
                self.illness_period -= 1
                self.try_to_die_after_one_day()
                if self.state != 4 and self.illness_period <= 0:
                    self.go_into_recovery_today = True

        self.do_shopping()


class CashierAgent(Agent):
    def __init__(self, unique_id, model, neighbourhood_id):
        super().__init__(unique_id, model)
        self.neighbourhood_id = neighbourhood_id

        self.go_into_incubation_today = False
        self.go_into_prodromal_today = False
        self.replace_cashier_today = False

        if self.model.start_with_infected_cashiers_only:
            self.state = 1
            self.model.incubation_cashiers += 1
            self.incubation_period = 1
            self.prodromal_period = 0
            self.illness_period = 0
        else:
            # initial infection with given probability
            if np.random.rand() > model.initial_infection_probability:
                self.state = 0
                self.model.susceptible_cashiers += 1
                self.incubation_period = 0
                self.prodromal_period = 0
                self.illness_period = 0
            else:
                self.state = 1
                self.model.incubation_cashiers += 1
                self.incubation_period = get_incubation_period(model=self.model)
                self.prodromal_period = 0
                self.illness_period = 0

    # Helpful agent functions =========================================================================================
    def set_all_disease_periods_to_zero(self):
        self.incubation_period = 0
        self.prodromal_period = 0
        self.illness_period = 0

    def go_into_susceptible_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 0
        self.model.susceptible_cashiers += 1

    def go_into_incubation_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 1
        self.model.susceptible_cashiers -= 1
        self.model.incubation_cashiers += 1
        self.incubation_period = get_incubation_period(model=self.model)

    def go_into_prodromal_phase(self):
        self.set_all_disease_periods_to_zero()
        self.state = 2
        self.model.incubation_cashiers -= 1
        self.model.prodromal_cashiers += 1
        self.prodromal_period = get_prodromal_period(model=self.model)

    def replace_cashier(self):
        self.model.replaced_cashiers += 1
        self.model.prodromal_cashiers -= 1
        self.go_into_susceptible_phase()

    def update_state(self):
        if self.go_into_incubation_today:
            self.go_into_incubation_phase()
            self.go_into_incubation_today = False
        elif self.go_into_prodromal_today:
            self.go_into_prodromal_phase()
            self.go_into_prodromal_today = False
        elif self.replace_cashier_today:
            self.replace_cashier()
            self.replace_cashier_today = False

    # TEST ===========================================================================================================
    def show_neighbourhood(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        return self.pos, possible_steps

    # Cashier STEP ====================================================================================================
    def step(self):
        if self.state == 1:
            self.incubation_period -= 1
            if self.incubation_period <= 0:
                self.go_into_prodromal_today = True

        elif self.state == 2:
            self.prodromal_period -= 1
            if self.prodromal_period <= 0:
                self.replace_cashier_today = True

