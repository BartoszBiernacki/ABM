from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from disease_spread_model.mesa_modified.mesa_time_modified import \
    OrderedActivation

from .my_math_utils import *

from .disease_agent import CashierAgent
from .my_model_utils import \
    create_array_of_shopping_days_for_each_household_for_each_week
from .my_model_utils import \
    create_array_of_extra_shopping_days_for_each_household_for_each_week
from .my_model_utils import neigh_id_of_extra_shopping
from .my_model_utils import make_ordinary_shopping_core
from .my_model_utils import make_extra_shopping_core
from .my_model_utils import update_A_states_core
from .my_model_utils import update_C_states_core
from .my_model_utils import find_out_who_wants_to_do_extra_shopping
from .my_model_utils import find_out_who_wants_to_do_shopping
from .my_model_utils import set_on_shopping_true_depends_on_shopping_situation
from .my_model_utils import \
    set_on_extra_shopping_true_depends_on_shopping_situation
from .my_model_utils import find_neighbouring_neighbourhoods
from .my_model_utils import try_to_infect_housemates_core
from .my_model_utils import get_incubation_period
from .my_model_utils import get_prodromal_period

# Ordinary population
from .collectors import calculate_ordinary_pearson_number_susceptible
from .collectors import calculate_ordinary_pearson_number_incubation
from .collectors import calculate_ordinary_pearson_number_prodromal
from .collectors import calculate_ordinary_pearson_number_illness
from .collectors import calculate_ordinary_pearson_number_illness_visible
from .collectors import calculate_ordinary_pearson_number_illness_invisible
from .collectors import calculate_ordinary_pearson_number_dead
from .collectors import calculate_ordinary_pearson_number_recovery

from .collectors import calculate_infected_toll

# Cashiers population
from .collectors import calculate_incubation_cashiers
from .collectors import calculate_prodromal_cashiers
from .collectors import calculate_susceptible_cashiers
from .collectors import calculate_replaced_cashiers

# Ordinary shopping
from .collectors import calculate_ordinary_customers
from .collectors import calculate_susceptible_customers
from .collectors import calculate_incubation_customers
from .collectors import calculate_prodromal_customers
from .collectors import calculate_illness_customers
from .collectors import calculate_recovery_customers
from .collectors import calculate_infected_by_self_cashier_today

# Extra shopping
from .collectors import calculate_extra_customers
from .collectors import calculate_extra_susceptible_customers
from .collectors import calculate_extra_incubation_customers
from .collectors import calculate_extra_prodromal_customers
from .collectors import calculate_extra_illness_customers
from .collectors import calculate_extra_recovery_customers
from .collectors import calculate_infected_by_extra_cashier_today

# Others
from .collectors import get_day


class DiseaseModel(Model):
    def __init__(self,
                 grid_size: tuple[int],
                 N: int,
                 customers_in_household: int,
                 beta: float,
                 mortality: float,
                 visibility: float,

                 avg_incubation_period, incubation_period_bins,
                 avg_prodromal_period, prodromal_period_bins,
                 avg_illness_period, illness_period_bins,

                 infected_cashiers_at_start,
                 percent_of_infected_customers_at_start,
                 extra_shopping_boolean,
                 housemate_infection_probability,
                 max_steps,
                 ):  # sourcery no-metrics
        super().__init__()
        self.running = True  # required for BatchRunner

        # Applying constructor arguments as self properties ****************
        self.width = grid_size[0]
        self.height = grid_size[1]
        self.num_of_households_in_neighbourhood = N
        self.customers_in_household = customers_in_household

        self.avg_incubation_period = avg_incubation_period
        self.incubation_period_bins = incubation_period_bins
        self.avg_prodromal_period = avg_prodromal_period
        self.prodromal_period_bins = prodromal_period_bins
        self.avg_illness_period = avg_illness_period
        self.illness_period_bins = illness_period_bins

        self.beta = beta
        self.mortality = mortality
        self.visibility = visibility

        self.infected_cashiers_at_start = infected_cashiers_at_start
        self.percent_of_infected_customers_at_start = \
            percent_of_infected_customers_at_start

        self.extra_shopping_boolean = extra_shopping_boolean
        self.housemate_infection_probability = housemate_infection_probability
        self.max_steps = max_steps
        # **************************************************************

        self.current_id = 0
        self.day = 0

        # Helpful attributes ******************************************
        self.total_neighbourhoods = int(self.width * self.height)
        self.total_households = int(self.num_of_households_in_neighbourhood *
                                    self.total_neighbourhoods)
        self.number_of_weeks_to_make_shopping_everywhere = nearest_neighbours(
            y=self.height, x=self.width)
        self.max_weeks = self.max_steps // 7 + 1
        self.cashiers_by_neigh = np.empty(
            self.total_neighbourhoods, dtype=CashierAgent)
        self.neigh_id_by_house_id = np.empty(
            self.total_households, dtype=np.int8)
        self.suspicious_households = np.zeros(
            self.total_households, dtype=bool)
        self.neigh_id_of_infected_cashiers_at_start = random.sample(
            range(self.total_neighbourhoods), self.infected_cashiers_at_start)
        # ******************************************************************

        # Variety of disease stages duration ********************************
        # Incubation
        S, exponents = get_S_and_exponents_for_sym_hist(
            bins=self.incubation_period_bins)
        self.S_incubation = S
        self.exponents_incubation = exponents
        # Prodromal
        S, exponents = get_S_and_exponents_for_sym_hist(
            bins=self.prodromal_period_bins)
        self.S_prodromal = S
        self.exponents_prodromal = exponents
        # Illness
        S, exponents = get_S_and_exponents_for_sym_hist(
            bins=self.illness_period_bins)
        self.S_illness = S
        self.exponents_illness = exponents
        # ************************************************************

        # Model necessary attributes *********************************
        self.grid = MultiGrid(width=self.width, height=self.height, torus=True)
        self.schedule = OrderedActivation(self)
        # ************************************************************

        # SETTING SHOPPING DAYS FOR EACH HOUSEHOLD ******************
        # [week][household_id] --> [day1, day2],
        # day1 != day2, day in {0, 1, 2, 3, 4, 5, 6}
        weeks = int(self.max_steps // 7 + 1)
        shopping_days_in_week = 2
        self.shopping_days_for_each_household_for_each_week = \
            create_array_of_shopping_days_for_each_household_for_each_week(
                array_to_fill=np.empty(
                    (weeks, self.total_households, shopping_days_in_week),
                    dtype=np.int8))
        # *****************************************************************

        # Translation [x, y] --> neigh_id and reversed ******************
        # [y, x] --> neighbourhood_id
        self.neighbourhood_yx_position_to_id = np.empty(
            (self.height, self.width), dtype=np.int8)
        #  neighbourhood_id --> [y, x]
        self.neighbourhood_id_to_yx_position = np.empty(
            (self.total_neighbourhoods, 2), dtype=np.int8)

        # [neighbourhood_id] --> 0 or 1 or 2
        self.C_state_by_neigh_id = np.empty(
            self.total_neighbourhoods, dtype=np.int8)
        self.C_incubation_duration_by_neigh_id = np.zeros(
            self.total_neighbourhoods, dtype=np.int8)
        self.C_prodromal_duration_by_neigh_id = np.zeros(
            self.total_neighbourhoods, dtype=np.int8)
        # *****************************************************************

        # Agent states and corespondent times ****************************
        # [household_id] --> [int_1, ..., int_N],
        # N=num_of_customers_in_household, int_m in {-1, 0, 1, 2, 3, 4}
        self.A_state_by_house_id = np.empty(
            (self.total_households, self.customers_in_household),
            dtype=np.int8)

        # 'A_ignore_quarantine_by_house_id' does agent have only hidden
        # disease symptoms, which mean he can always go shopping and never die?
        self.A_ignore_quarantine_by_house_id = np.random.rand(
            self.total_households, self.customers_in_household) \
                                               > self.visibility

        self.A_incubation_duration_by_house_id = np.empty_like(
            self.A_state_by_house_id, dtype=np.int8)
        self.A_prodromal_duration_by_house_id = np.empty_like(
            self.A_state_by_house_id, dtype=np.int8)
        self.A_illness_duration_by_house_id = np.empty_like(
            self.A_state_by_house_id, dtype=np.int8)
        # ************************************************************

        # Agent (extra) shopping T/F and is agent was infected on today's
        # (extra) shopping ************************************************
        self.A_on_shopping_by_house_id = np.zeros_like(
            self.A_state_by_house_id, dtype=bool)
        self.A_on_extra_shopping_by_house_id = np.zeros_like(
            self.A_state_by_house_id, dtype=bool)

        self.A_go_incubation_because_shopping = np.zeros_like(
            self.A_state_by_house_id, dtype=bool)
        self.A_go_incubation_because_housemate = np.zeros_like(
            self.A_state_by_house_id, dtype=bool)
        self.C_go_incubation_because_shopping = np.zeros_like(
            self.C_state_by_neigh_id, dtype=bool)
        # *********************************************************

        # COUNTERS ***********************************************
        self.replaced_cashiers = 0

        self.ordinary_customers = 0
        self.susceptible_customers = 0
        self.incubation_customers = 0
        self.prodromal_customers = 0
        self.illness_customers = 0
        self.recovery_customers = 0
        self.infected_customers_by_self_cashier = 0

        self.extra_customers = 0
        self.extra_susceptible_customers = 0
        self.extra_incubation_customers = 0
        self.extra_prodromal_customers = 0
        self.extra_illness_customers = 0
        self.extra_recovery_customers = 0
        self.infected_customers_by_extra_cashier = 0
        # *************************************************************

        # AGENTS CREATION ********************************************
        neighbourhood_id = 0
        household_id = 0
        for y in range(self.height):
            for x in range(self.width):
                # Create customers for given neighbourhood
                for _ in range(self.num_of_households_in_neighbourhood):
                    for household_member_id in range(
                            self.customers_in_household):
                        # create incubated or healthy client at stert
                        if random.random() < self.percent_of_infected_customers_at_start / 100:
                            self.A_state_by_house_id[household_id][
                                household_member_id] = 1
                            self.A_incubation_duration_by_house_id[
                                household_id][household_member_id] = \
                                get_incubation_period(
                                    avg_incubation_period=self.avg_incubation_period,
                                    incubation_period_bins=self.incubation_period_bins,
                                    S_incubation=self.S_incubation,
                                    exponents_incubation=self.exponents_incubation)
                        else:
                            self.A_state_by_house_id[household_id][
                                household_member_id] = 0
                            self.A_incubation_duration_by_house_id[
                                household_id][household_member_id] = 0

                        self.A_prodromal_duration_by_house_id[household_id][
                            household_member_id] = 0
                        self.A_illness_duration_by_house_id[household_id][
                            household_member_id] = 0

                        self.A_on_shopping_by_house_id[household_id][
                            household_member_id] = False
                        self.A_on_extra_shopping_by_house_id[household_id][
                            household_member_id] = False

                    self.neigh_id_by_house_id[household_id] = neighbourhood_id
                    household_id += 1

                # Create cashier for given neighbourhood
                if neighbourhood_id in self.neigh_id_of_infected_cashiers_at_start:
                    self.C_state_by_neigh_id[neighbourhood_id] = 2
                    self.C_prodromal_duration_by_neigh_id[neighbourhood_id] = \
                        get_prodromal_period(
                            avg_prodromal_period=self.avg_prodromal_period,
                            prodromal_period_bins=self.prodromal_period_bins,
                            S_prodromal=self.S_prodromal,
                            exponents_prodromal=self.exponents_prodromal)
                else:
                    self.C_state_by_neigh_id[neighbourhood_id] = 0

                b = CashierAgent(unique_id=self.next_id(), model=self,
                                 neighbourhood_id=neighbourhood_id)
                self.schedule.add(b)
                self.grid.place_agent(agent=b, pos=(x, y))
                self.cashiers_by_neigh[neighbourhood_id] = b

                self.neighbourhood_yx_position_to_id[y][x] = neighbourhood_id
                self.neighbourhood_id_to_yx_position[
                    neighbourhood_id] = np.array([y, x])
                neighbourhood_id += 1
        # ************************************************************************************************************

        if self.extra_shopping_boolean:
            if self.number_of_weeks_to_make_shopping_everywhere:
                # [week][household_id] --> int1, possible ints = {0, 1, 2, 3, 4, 5, 6}
                self.extra_shopping_days = create_array_of_extra_shopping_days_for_each_household_for_each_week(
                    array_to_fill=np.empty(
                        (self.max_weeks, self.total_households, 1),
                        dtype=np.int8))

                # [neighbourhood_id] --> [neighbour_neighbourhood_1_id, ..., neighbour_neighbourhood_n_id], n in {0, 1, 2, 3, 4}
                self.nearest_neighbourhoods = find_neighbouring_neighbourhoods(
                    all_cashiers=self.cashiers_by_neigh,
                    neighbourhood_pos_to_id=self.neighbourhood_yx_position_to_id)

                # [week][household_id] --> neighbourhood_id to visit
                array_to_fill = np.ones(
                    (self.max_weeks, self.total_households), dtype=np.int8) * (
                                    -1)
                self.neigh_id_of_extra_shopping_by_week_and_house_id = \
                    neigh_id_of_extra_shopping(
                        nearest_neighbourhoods_by_neigh_id=self.nearest_neighbourhoods,
                        neigh_id_by_house_id=self.neigh_id_by_house_id,
                        array_to_fill=array_to_fill)
            else:
                self.extra_shopping_boolean = False

        self.datacollector = DataCollector(
            model_reporters={"Day": get_day,

                             "Susceptible people": calculate_ordinary_pearson_number_susceptible,
                             "Incubation people": calculate_ordinary_pearson_number_incubation,
                             "Prodromal people": calculate_ordinary_pearson_number_prodromal,
                             "Illness people": calculate_ordinary_pearson_number_illness,
                             "Illness visible people": calculate_ordinary_pearson_number_illness_visible,
                             "Illness invisible people": calculate_ordinary_pearson_number_illness_invisible,
                             "Dead people": calculate_ordinary_pearson_number_dead,
                             "Recovery people": calculate_ordinary_pearson_number_recovery,

                             "Infected toll": calculate_infected_toll,

                             "Susceptible cashiers": calculate_susceptible_cashiers,
                             "Incubation cashiers": calculate_incubation_cashiers,
                             "Prodromal cashiers": calculate_prodromal_cashiers,
                             "Replaced cashiers": calculate_replaced_cashiers,

                             "Ordinary customers": calculate_ordinary_customers,
                             "Susceptible customers": calculate_susceptible_customers,
                             "Incubation customers": calculate_incubation_customers,
                             "Prodromal customers": calculate_prodromal_customers,
                             "Illness customers": calculate_illness_customers,
                             "Recovery customers": calculate_recovery_customers,
                             "Infected by self cashier": calculate_infected_by_self_cashier_today,

                             "Extra customers": calculate_extra_customers,
                             "Susceptible extra customers": calculate_extra_susceptible_customers,
                             "Incubation extra customers": calculate_extra_incubation_customers,
                             "Prodromal extra customers": calculate_extra_prodromal_customers,
                             "Illness extra customers": calculate_extra_illness_customers,
                             "Recovery extra customers": calculate_extra_recovery_customers,
                             "Infected by extra cashier": calculate_infected_by_extra_cashier_today})

    # ----------------------------------------------------------------------------------------------------------------
    def make_shopping_decisions_about_self_neighbourhood(self):
        # situation[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
        situation = find_out_who_wants_to_do_shopping(
            shopping_days_for_each_household_for_each_week=self.shopping_days_for_each_household_for_each_week,
            day=self.day,
            agents_state_grouped_by_households=self.A_state_by_house_id,
            A_ignore_quarantine_by_house_id=self.A_ignore_quarantine_by_house_id,
            array_to_fill=np.zeros((self.total_households, 2), dtype=np.int8))

        # ordinary shopping
        set_on_shopping_true_depends_on_shopping_situation(
            total_num_of_households=self.total_households,
            situation=situation,
            array_to_fill=self.A_on_shopping_by_house_id)

    def make_extra_shopping_decisions_about_neighbouring_neighbourhoods(self):
        situation = find_out_who_wants_to_do_extra_shopping(
            extra_shopping_days=self.extra_shopping_days,
            day=self.day,
            agents_state_grouped_by_households=self.A_state_by_house_id,
            A_ignore_quarantine_by_house_id=self.A_ignore_quarantine_by_house_id,
            array_to_fill=np.zeros((self.total_households, 2), dtype=np.int8))

        # extra shopping
        set_on_extra_shopping_true_depends_on_shopping_situation(
            total_num_of_households=self.total_households,
            situation=situation,
            array_to_fill=self.A_on_extra_shopping_by_house_id)

    def make_ordinary_shopping(self):
        details = make_ordinary_shopping_core(
            total_households=self.total_households,
            num_of_customers_in_household=self.customers_in_household,
            A_on_shopping_by_house_id=self.A_on_shopping_by_house_id,
            A_state_by_house_id=self.A_state_by_house_id,
            C_state_by_neigh_id=self.C_state_by_neigh_id,
            neigh_id_by_house_id=self.neigh_id_by_house_id,
            beta=self.beta,
            A_go_incubation_because_shopping=self.A_go_incubation_because_shopping,
            C_go_incubation_because_shopping=self.C_go_incubation_because_shopping)

        # customers, susceptible_customers, incubation_customers, prodromal_customers,
        # illness_customers, recovery_customers, infected_by_self_cashier
        self.ordinary_customers = details[0]
        self.susceptible_customers = details[1]
        self.incubation_customers = details[2]
        self.prodromal_customers = details[3]
        self.illness_customers = details[4]
        self.recovery_customers = details[5]
        self.infected_customers_by_self_cashier = details[6]

    def make_extra_shopping(self):
        # print(self.neigh_id_of_extra_shopping_by_week_and_house_id)

        details = make_extra_shopping_core(
            total_households=self.total_households,
            num_of_customers_in_household=self.customers_in_household,
            A_on_extra_shopping_by_house_id=self.A_on_extra_shopping_by_house_id,
            A_state_by_house_id=self.A_state_by_house_id,
            C_state_by_neigh_id=self.C_state_by_neigh_id,
            neigh_id_by_house_id=self.neigh_id_by_house_id,
            beta=self.beta,
            A_go_incubation_because_shopping=self.A_go_incubation_because_shopping,
            C_go_incubation_because_shopping=self.C_go_incubation_because_shopping,
            neigh_id_of_extra_shopping_by_week_and_house_id=self.neigh_id_of_extra_shopping_by_week_and_house_id,
            day=self.day)

        self.extra_customers = details[0]
        self.extra_susceptible_customers = details[1]
        self.extra_incubation_customers = details[2]
        self.extra_prodromal_customers = details[3]
        self.extra_illness_customers = details[4]
        self.extra_recovery_customers = details[5]
        self.infected_customers_by_extra_cashier = details[6]

    def make_shopping(self):
        self.make_shopping_decisions_about_self_neighbourhood()
        self.make_ordinary_shopping()
        if self.extra_shopping_boolean:
            self.make_extra_shopping_decisions_about_neighbouring_neighbourhoods()
            self.make_extra_shopping()

    def try_to_infect_housemates(self):
        try_to_infect_housemates_core(total_households=self.total_households,
                                      num_of_customers_in_household=self.customers_in_household,
                                      suspicious_households=self.suspicious_households,
                                      A_state_by_house_id=self.A_state_by_house_id,
                                      infection_probability=self.housemate_infection_probability,
                                      A_go_incubation_because_housemate=self.A_go_incubation_because_housemate)

    def set_shopping_stats_to_zero(self):
        self.susceptible_customers = 0
        self.incubation_customers = 0
        self.prodromal_customers = 0
        self.illness_customers = 0
        self.recovery_customers = 0
        self.infected_customers_by_self_cashier = 0

        self.extra_customers = 0
        self.extra_susceptible_customers = 0
        self.extra_incubation_customers = 0
        self.extra_prodromal_customers = 0
        self.extra_illness_customers = 0
        self.extra_recovery_customers = 0
        self.infected_customers_by_extra_cashier = 0

    def update_A_states(self):
        update_A_states_core(total_households=self.total_households,
                             num_of_customers_in_household=self.customers_in_household,
                             A_state_by_house_id=self.A_state_by_house_id,
                             suspicious_households=self.suspicious_households,
                             A_go_incubation_because_shopping=self.A_go_incubation_because_shopping,
                             A_go_incubation_because_housemate=self.A_go_incubation_because_housemate,
                             A_incubation_duration_by_house_id=self.A_incubation_duration_by_house_id,
                             A_prodromal_duration_by_house_id=self.A_prodromal_duration_by_house_id,
                             A_illness_duration_by_house_id=self.A_illness_duration_by_house_id,
                             mortality=self.mortality,
                             A_ignore_quarantine_by_house_id=self.A_ignore_quarantine_by_house_id,
                             avg_incubation_period=self.avg_incubation_period,
                             incubation_period_bins=self.incubation_period_bins,
                             S_incubation=self.S_incubation,
                             exponents_incubation=self.exponents_incubation,
                             avg_prodromal_period=self.avg_prodromal_period,
                             prodromal_period_bins=self.prodromal_period_bins,
                             S_prodromal=self.S_prodromal,
                             exponents_prodromal=self.exponents_prodromal,
                             avg_illness_period=self.avg_illness_period,
                             illness_period_bins=self.illness_period_bins,
                             S_illness=self.S_illness,
                             exponents_illness=self.exponents_illness)

    def update_C_states(self):
        replaced_today = update_C_states_core(
            total_neighbourhoods=self.total_neighbourhoods,
            C_state_by_neigh_id=self.C_state_by_neigh_id,
            C_go_incubation_because_shopping=self.C_go_incubation_because_shopping,
            C_incubation_duration_by_neigh_id=self.C_incubation_duration_by_neigh_id,
            C_prodromal_duration_by_neigh_id=self.C_prodromal_duration_by_neigh_id,
            avg_incubation_period=self.avg_incubation_period,
            incubation_period_bins=self.incubation_period_bins,
            S_incubation=self.S_incubation,
            exponents_incubation=self.exponents_incubation,
            avg_prodromal_period=self.avg_prodromal_period,
            prodromal_period_bins=self.prodromal_period_bins,
            S_prodromal=self.S_prodromal,
            exponents_prodromal=self.exponents_prodromal)

        self.replaced_cashiers += replaced_today

    def update_agents_state(self):
        self.update_A_states()
        self.update_C_states()

    def release_memory(self):
        self.num_of_households_in_neighbourhood = None
        self.customers_in_household = None
        self.avg_incubation_period = None
        self.incubation_period_bins = None
        self.avg_prodromal_period = None
        self.prodromal_period_bins = None
        self.avg_illness_period = None
        self.illness_period_bins = None
        self.beta = None
        self.mortality = None
        self.infected_cashiers_at_start = None
        self.percent_of_infected_customers_at_start = None
        self.extra_shopping_boolean = None
        self.housemate_infection_probability = None
        self.max_steps = None
        self.current_id = None
        self.day = None

        self.total_neighbourhoods = None
        self.total_households = None
        self.number_of_weeks_to_make_shopping_everywhere = None
        self.cashiers_by_neigh = None
        self.neigh_id_by_house_id = None
        self.suspicious_households = None
        self.neigh_id_of_infected_cashiers_at_start = None

        self.S_incubation = None
        self.exponents_incubation = None
        self.S_prodromal = None
        self.exponents_prodromal = None
        self.S_illness = None
        self.exponents_illness = None

        self.grid = None
        self.schedule = None

        self.shopping_days_for_each_household_for_each_week = None
        self.neighbourhood_yx_position_to_id = None
        self.neighbourhood_id_to_yx_position = None

        self.C_state_by_neigh_id = None
        self.C_incubation_duration_by_neigh_id = None
        self.C_prodromal_duration_by_neigh_id = None

        self.A_state_by_house_id = None
        self.A_incubation_duration_by_house_id = None
        self.A_prodromal_duration_by_house_id = None
        self.A_illness_duration_by_house_id = None
        self.A_on_shopping_by_house_id = None
        self.A_on_extra_shopping_by_house_id = None
        self.A_go_incubation_because_shopping = None
        self.A_go_incubation_because_housemate = None
        self.C_go_incubation_because_shopping = None

        self.replaced_cashiers = None
        self.ordinary_customers = None
        self.susceptible_customers = None
        self.incubation_customers = None
        self.prodromal_customers = None
        self.illness_customers = None
        self.recovery_customers = None
        self.infected_customers_by_self_cashier = None
        self.extra_customers = None
        self.extra_susceptible_customers = None
        self.extra_incubation_customers = None
        self.extra_prodromal_customers = None
        self.extra_illness_customers = None
        self.extra_recovery_customers = None
        self.infected_customers_by_extra_cashier = None

    def step(self):

        self.set_shopping_stats_to_zero()
        self.make_shopping()

        if self.housemate_infection_probability > 0:
            self.try_to_infect_housemates()

        self.schedule.step()
        self.datacollector.collect(model=self)

        if self.day + 1 == self.max_steps:
            self.release_memory()
            self.running = False
        else:
            self.update_agents_state()
            self.day += 1
