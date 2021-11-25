from mesa import Model
from mesa.time import RandomActivation
from mesa_time_modified import OrderedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from my_math_utils import *

from disease_agent import OrdinaryPearsonAgent, CashierAgent
from my_model_utils import create_array_of_shopping_days_for_each_household_for_each_week, \
    find_out_who_wants_to_do_extra_shopping, get_day
from my_model_utils import find_out_who_wants_to_do_shopping
from my_model_utils import calculate_ordinary_pearson_number_susceptible
from my_model_utils import calculate_ordinary_pearson_number_incubation
from my_model_utils import calculate_ordinary_pearson_number_prodromal
from my_model_utils import calculate_ordinary_pearson_number_illness
from my_model_utils import calculate_ordinary_pearson_number_dead
from my_model_utils import calculate_ordinary_pearson_number_recovery
from my_model_utils import calculate_susceptible_customers
from my_model_utils import calculate_incubation_customers
from my_model_utils import calculate_prodromal_customers
from my_model_utils import calculate_recovery_customers
from my_model_utils import calculate_extra_customers
from my_model_utils import calculate_incubation_cashiers
from my_model_utils import calculate_prodromal_cashiers
from my_model_utils import calculate_susceptible_cashiers
from my_model_utils import create_order_of_shopping_in_neighbouring_neighbourhoods
from my_model_utils import find_neighbouring_neighbourhoods

from my_model_utils import calculate_infected_customers_by_cashier_today


class DiseaseModel(Model):
    def __init__(self,  width, height, num_of_households_in_neighbourhood, num_of_customers_in_household,
                 num_of_cashiers_in_neighbourhood,
                 beta, avg_incubation_period, avg_prodromal_period, avg_illness_period, mortality, die_at_once,
                 initial_infection_probability,
                 start_with_infected_cashiers_only,
                 random_ordinary_pearson_activation,
                 extra_shopping_boolean,
                 max_steps):
        super().__init__()
        self.running = True  # required for BatchRunner
        self.start_with_infected_cashiers_only = start_with_infected_cashiers_only
        self.num_of_households_in_neighbourhood = num_of_households_in_neighbourhood
        self.num_of_customers_in_household = num_of_customers_in_household
        self.num_of_cashiers_in_neighbourhood = num_of_cashiers_in_neighbourhood
        self.beta = beta
        self.avg_incubation_period = avg_incubation_period
        self.avg_prodromal_period = avg_prodromal_period
        self.avg_illness_period = avg_illness_period
        self.mortality = mortality
        self.die_at_once = die_at_once
        self.probability_of_staying_alive_after_one_day_of_illness = (1 - self.mortality) ** (1 / self.avg_illness_period)
        self.initial_infection_probability = initial_infection_probability
        self.extra_shopping_boolean = extra_shopping_boolean
        self.max_steps = max_steps

        self.current_id = 0
        self.day = 0

        # model necessary attributes *********************************************************************************
        self.grid = MultiGrid(width=width, height=height, torus=True)
        if random_ordinary_pearson_activation:
            self.schedule = RandomActivation(self)
        else:
            self.schedule = OrderedActivation(self)
        # ************************************************************************************************************

        # SETTING SHOPPING DAYS FOR EACH HOUSEHOLD *******************************************************************
        # [week][household_id] --> [int1, int2], int1 != int2, possible ints = {0, 1, 2, 3, 4, 5, 6}
        self.shopping_days_for_each_household_for_each_week = \
            create_array_of_shopping_days_for_each_household_for_each_week(
                array_to_fill=np.empty((self.max_steps // 7 + 1, self.num_of_households_in_neighbourhood * width *
                                        height, 2), dtype=np.int8),
                days_array=np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8))
        # ************************************************************************************************************

        self.total_num_of_households = width * height * num_of_households_in_neighbourhood

        # [y, x] --> neighbourhood_id
        self.neighbourhood_yx_position_to_id = np.empty((height, width), dtype=np.int8)

        #  neighbourhood_id --> [y, x]
        self.neighbourhood_id_to_yx_position = np.empty((height*width, 2), dtype=np.int8)

        # [neighbourhood_id] --> [cashier1, ..., cashierN], N=num od cashiers in given neighbourhood
        self.cashiers_grouped_by_neighbourhood_id = np.empty((height*width, self.num_of_cashiers_in_neighbourhood),
                                                             dtype=CashierAgent)

        # [household_id] --> [OrdinaryPearson_1, ..., OrdinaryPearson_N], N=num_of_customers_in_household
        self.customers_grouped_by_households = np.empty((self.total_num_of_households, num_of_customers_in_household),
                                                        dtype=OrdinaryPearsonAgent)

        # [household_id] --> [int1, ..., intK], int_m=customer[household_id][m].state, K=num of agents in each household
        self.agents_state_grouped_by_households = np.empty((self.total_num_of_households,
                                                           self.num_of_customers_in_household), dtype=np.int8)

        # [neighbourhood_id] --> [neighbour neighbourhood_1_id, ..., neighbour neighbourhood_k_id], k in {0, 1, 2, 3, 4}
        self.neighbouring_neighbourhoods_grouped_by_neighbourhood_id = \
            np.zeros((height*width, get_number_of_cell_neighbours(y=height, x=width)))

        # COUNTERS ***************************************************************************************************
        self.ordinary_pearson_number_susceptible = 0
        self.ordinary_pearson_number_incubation = 0
        self.ordinary_pearson_number_prodromal = 0
        self.ordinary_pearson_number_illness = 0
        self.ordinary_pearson_number_recovery = 0
        self.ordinary_pearson_number_dead = 0

        self.customers_infected_by_cashier_today = 0
        self.cashiers_infected_by_customers_today = 0

        self.susceptible_cashiers = 0
        self.incubation_cashiers = 0
        self.prodromal_cashiers = 0
        self.replaced_cashiers = 0

        self.susceptible_customers = 0
        self.incubation_customers = 0
        self.prodromal_customers = 0
        self.recovery_customers = 0

        self.extra_customers = 0
        self.susceptible_extra_customers = 0
        self.incubation_extra_customers = 0
        self.prodromal_extra_customers = 0
        self.recovery_extra_customers = 0
        # ************************************************************************************************************

        # AGENTS CREATION ********************************************************************************************
        neighbourhood_id = 0
        household_id = 0
        for y in range(height):
            for x in range(width):
                household_number = 0
                # Create customers for given neighbourhood
                for _ in range(self.num_of_households_in_neighbourhood):
                    for household_member in range(self.num_of_customers_in_household):
                        a = OrdinaryPearsonAgent(unique_id=self.next_id(), model=self,
                                                 number_in_household=household_member,  household_id=household_id,
                                                 neighbourhood_id=neighbourhood_id)
                        self.schedule.add(a)
                        self.grid.place_agent(agent=a, pos=(x, y))
                        self.customers_grouped_by_households[household_id][household_member] = a
                        self.agents_state_grouped_by_households[household_id][household_member] = a.state
                    household_id += 1
                    household_number += 1
                # Create cashiers for given neighbourhood
                for i in range(self.num_of_cashiers_in_neighbourhood):
                    b = CashierAgent(unique_id=self.next_id(), model=self, neighbourhood_id=neighbourhood_id)
                    self.schedule.add(b)
                    self.grid.place_agent(agent=b, pos=(x, y))
                    self.cashiers_grouped_by_neighbourhood_id[neighbourhood_id][i] = b
                self.neighbourhood_yx_position_to_id[y][x] = neighbourhood_id
                self.neighbourhood_id_to_yx_position[neighbourhood_id] = np.array([y, x])
                neighbourhood_id += 1
        # ************************************************************************************************************



        self.needed_number_of_weeks_to_make_shopping_everywhere = get_number_of_cell_neighbours(y=height, x=width)

        if self.needed_number_of_weeks_to_make_shopping_everywhere:

            # [week][household_id] --> int1, possible ints = {0, 1, 2, 3, 4, 5, 6}
            self.extra_shopping_days_for_each_household_for_each_week = \
                create_array_of_shopping_days_for_each_household_for_each_week(
                    array_to_fill=np.empty((self.max_steps // 7 + 1, self.num_of_households_in_neighbourhood * width *
                                            height, 1), dtype=np.int8),
                    days_array=np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8))

            # [neighbourhood_id] --> [neighbour_neighbourhood_1_id, ..., neighbour_neighbourhood_n_id], n in {0, 1, 2, 3, 4}
            self.neighbouring_neighbourhoods = find_neighbouring_neighbourhoods(
                all_cashiers=self.cashiers_grouped_by_neighbourhood_id,
                neighbourhood_pos_to_id=self.neighbourhood_yx_position_to_id)

            # [neighbourhood_id][num_of_specific_household_in_that_neighbourhood][week] --> neighbour to visit
            self.order_of_shopping_in_neighbouring_neighbourhoods =\
                create_order_of_shopping_in_neighbouring_neighbourhoods(
                    neighbouring_neighbourhoods_grouped_by_neighbourhood_id=self.neighbouring_neighbourhoods,
                    array_to_fill=np.empty((width*height, self.num_of_households_in_neighbourhood, self.max_steps // 7 + 1),
                                           dtype=np.int8))
        else:
            self.extra_shopping_boolean = False

        self.datacollector = DataCollector(
            model_reporters={"Day": get_day,
                             "Susceptible people": calculate_ordinary_pearson_number_susceptible,
                             "Incubation people": calculate_ordinary_pearson_number_incubation,
                             "Prodromal people": calculate_ordinary_pearson_number_prodromal,
                             "Illness people": calculate_ordinary_pearson_number_illness,
                             "Dead people": calculate_ordinary_pearson_number_dead,
                             "Recovery people": calculate_ordinary_pearson_number_recovery,
                             "Susceptible customers":  calculate_susceptible_customers,
                             "Incubation customers":  calculate_incubation_customers,
                             "Prodromal customers":  calculate_prodromal_customers,
                             "Recovery customers":  calculate_recovery_customers,
                             "Extra customers": calculate_extra_customers,
                             "Susceptible cashiers": calculate_susceptible_cashiers,
                             "Incubation cashiers": calculate_incubation_cashiers,
                             "Prodromal cashiers": calculate_prodromal_cashiers,
                             "Infected by cashier": calculate_infected_customers_by_cashier_today})

    # ----------------------------------------------------------------------------------------------------------------
    def make_shopping_decisions_about_self_neighbourhood(self):
        # situation[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
        situation = find_out_who_wants_to_do_shopping(
            shopping_days_for_each_household_for_each_week=self.shopping_days_for_each_household_for_each_week,
            day=self.day,
            agents_state_grouped_by_households=self.agents_state_grouped_by_households,
            array_to_fill=np.empty((self.total_num_of_households, 2), dtype=np.int8))

        # ordinary shopping
        for household_id in range(self.total_num_of_households):
            if situation[household_id][1]:
                volunteer_pos = situation[household_id][0]
                self.customers_grouped_by_households[household_id][volunteer_pos].on_shopping = True

    def make_extra_shopping_decisions_about_neighbouring_neighbourhoods(self):
        situation = find_out_who_wants_to_do_extra_shopping(
            extra_shopping_days_for_each_household_for_each_week=self.extra_shopping_days_for_each_household_for_each_week,
            day=self.day,
            agents_state_grouped_by_households=self.agents_state_grouped_by_households,
            array_to_fill=np.empty((self.total_num_of_households, 2), dtype=np.int8))

        # extra shopping
        for household_id in range(self.total_num_of_households):
            if situation[household_id][1]:
                volunteer_pos = situation[household_id][0]
                self.customers_grouped_by_households[household_id][volunteer_pos].on_extra_shopping = True

    def update_agents_state(self):
        for cashier in self.cashiers_grouped_by_neighbourhood_id.flat:
            cashier.update_state()

        for ordinary_agent in self.customers_grouped_by_households.flat:
            ordinary_agent.update_state()
    # -----------------------------------------------------------------------------------------------------------------

    def step(self):
        self.day += 1

        i = 0
        for ordinary_agent in self.customers_grouped_by_households.flat:
            if (ordinary_agent.asPD > 0 or ordinary_agent.asSC > 0) and ordinary_agent.did_shopping_today:
                i += 1
        # print()
        # print(f"DAY: {self.day-1}: {i}")
        for ordinary_agent in self.customers_grouped_by_households.flat:
            if (ordinary_agent.asPD > 0 or ordinary_agent.asSC > 0) and ordinary_agent.did_shopping_today:
                # print(f"{ordinary_agent.unique_id}, {ordinary_agent.asSC}, {ordinary_agent.asPD}, "
                #       f"{ordinary_agent.days}")
                ordinary_agent.did_shopping_today = False

        self.susceptible_customers = 0
        self.incubation_customers = 0
        self.prodromal_customers = 0
        self.recovery_customers = 0
        self.customers_infected_by_cashier_today = 0
        self.cashiers_infected_by_customers_today = 0
        self.extra_customers = 0

        self.make_shopping_decisions_about_self_neighbourhood()
        if self.extra_shopping_boolean:
            self.make_extra_shopping_decisions_about_neighbouring_neighbourhoods()

        self.schedule.step()

        self.datacollector.collect(model=self)
        self.update_agents_state()
