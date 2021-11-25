from my_math_utils import *


def get_day(model):
    return model.day


def get_incubation_period(model):
    return model.avg_incubation_period


def get_prodromal_period(model):
    return model.avg_prodromal_period


def get_illness_period(model):
    return model.avg_illness_period


def calculate_ordinary_pearson_number_susceptible(model):
    return model.ordinary_pearson_number_susceptible


def calculate_ordinary_pearson_number_incubation(model):
    return model.ordinary_pearson_number_incubation


def calculate_ordinary_pearson_number_prodromal(model):
    return model.ordinary_pearson_number_prodromal


def calculate_ordinary_pearson_number_illness(model):
    return model.ordinary_pearson_number_illness


def calculate_ordinary_pearson_number_dead(model):
    return model.ordinary_pearson_number_dead


def calculate_ordinary_pearson_number_recovery(model):
    return model.ordinary_pearson_number_recovery


def calculate_susceptible_customers(model):
    return model.susceptible_customers


def calculate_incubation_customers(model):
    return model.incubation_customers


def calculate_prodromal_customers(model):
    return model.prodromal_customers


def calculate_recovery_customers(model):
    return model.recovery_customers


def calculate_extra_customers(model):
    return model.extra_customers


def calculate_infected_customers_by_cashier_today(model):
    return model.customers_infected_by_cashier_today


def calculate_infected_cashiers_by_customers_today(model):
    return model.cashiers_infected_by_cashier_today


def calculate_susceptible_cashiers(model):
    return model.susceptible_cashiers


def calculate_incubation_cashiers(model):
    return model.incubation_cashiers


def calculate_prodromal_cashiers(model):
    return model.prodromal_cashiers


def calculate_replaced_cashiers(model):
    return model.repleaced.cashiers


def calculate_execution_time(model):
    return model.execution_time


def get_initial_available_shopping_household_members_dictionary(customers_grouped_by_households,
                                                                total_num_of_households):
    available_shopping_household_members = {}
    for household_id in range(total_num_of_households):
        household_members = customers_grouped_by_households[household_id]
        healthy_household_members = []
        for i, household_member in enumerate(household_members):
            if household_member.is_health_enough_to_do_shopping():
                healthy_household_members.append(household_member)
        available_shopping_household_members[household_id] = healthy_household_members
    return available_shopping_household_members


def find_neighbouring_neighbourhoods(all_cashiers, neighbourhood_pos_to_id):
    # [neighbourhood_id] --> [neighbour_neighbourhood_1_id, ..., neighbour_neighbourhood_n_id], n in {0, 1, 2, 3, 4}
    possible_steps_as_neighbourhood_ids = []
    for i, cashiers_in_neighbourhood in enumerate(all_cashiers):
        pos, possible_steps = cashiers_in_neighbourhood[0].show_neighbourhood()
        if pos in possible_steps:
            possible_steps.remove(pos)

        possible_steps_yx = [t[::-1] for t in possible_steps]
        possible_steps_for_specific_cell_as_neighbourhoods_id = []
        for possible_step_yx in possible_steps_yx:
            possible_steps_for_specific_cell_as_neighbourhoods_id.append(neighbourhood_pos_to_id[possible_step_yx])
        possible_steps_as_neighbourhood_ids.append(possible_steps_for_specific_cell_as_neighbourhoods_id)

    result = np.array(possible_steps_as_neighbourhood_ids)

    return result


@njit(cache=True)
def create_order_of_shopping_in_neighbouring_neighbourhoods(neighbouring_neighbourhoods_grouped_by_neighbourhood_id,
                                                            array_to_fill):
    # result[neighbourhood_id][num_of_household_in_that_neighbourhood][week] --> neighbouring neighbourhood to visit

    total_num_of_neighbourhoods, num_of_neighbouring_neighbourhoods = \
        neighbouring_neighbourhoods_grouped_by_neighbourhood_id.shape

    if num_of_neighbouring_neighbourhoods:  # do sth if there is at least one neighbour
        total_num_of_neighbourhoods, num_of_households_in_one_neighbourhood, num_of_weeks_to_simulate = np.shape(
            array_to_fill)
        num_of_needed_cycles = num_of_weeks_to_simulate // num_of_neighbouring_neighbourhoods + 1
        for neighbourhood_id in range(total_num_of_neighbourhoods):
            for household in range(num_of_households_in_one_neighbourhood):
                covered_weeks = 0
                for cycle in range(num_of_needed_cycles):
                    ran = np.random.permutation(neighbouring_neighbourhoods_grouped_by_neighbourhood_id[
                                                    neighbourhood_id])
                    for i in range(num_of_neighbouring_neighbourhoods):
                        array_to_fill[neighbourhood_id][household][covered_weeks] = ran[i]
                        covered_weeks += 1
                        if covered_weeks == num_of_weeks_to_simulate:
                            break
        return array_to_fill


@njit(cache=True)
def create_array_of_shopping_days_for_each_household_for_each_week(array_to_fill, days_array):
    total_num_of_weeks, total_num_of_households, num_of_shopping_days_in_week = array_to_fill.shape
    for week in range(total_num_of_weeks):
        for household in range(total_num_of_households):
            random.shuffle(days_array)
            array_to_fill[week][household] = days_array[: num_of_shopping_days_in_week]
    return array_to_fill


@njit(cache=True)
def find_out_who_wants_to_do_shopping(shopping_days_for_each_household_for_each_week,
                                      day,
                                      agents_state_grouped_by_households,
                                      array_to_fill):

    total_num_of_households, num_of_shopping_days_in_week = array_to_fill.shape

    day_mod_7 = day % 7
    week = day // 7
    # return[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
    for household_id in range(total_num_of_households):
        if day_mod_7 in shopping_days_for_each_household_for_each_week[week][household_id]:
            volunteer_pos = np.argmax(agents_state_grouped_by_households[household_id] <= 2)
            array_to_fill[household_id][0] = volunteer_pos

            if agents_state_grouped_by_households[household_id][volunteer_pos] <= 2:
                array_to_fill[household_id][1] = 1
        else:
            array_to_fill[household_id][1] = 0

    return array_to_fill


# @njit(cache=True)
def find_out_who_wants_to_do_extra_shopping(extra_shopping_days_for_each_household_for_each_week,
                                            day,
                                            agents_state_grouped_by_households,
                                            array_to_fill):

    total_num_of_households, num_of_shopping_days_in_week = array_to_fill.shape

    day_mod_7 = day % 7
    week = day // 7
    # return[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
    for household_id in range(total_num_of_households):
        if day_mod_7 in extra_shopping_days_for_each_household_for_each_week[week][household_id]:
            volunteer_pos = np.argmax(agents_state_grouped_by_households[household_id] <= 2)
            array_to_fill[household_id][0] = volunteer_pos

            if agents_state_grouped_by_households[household_id][volunteer_pos] <= 2:
                array_to_fill[household_id][1] = 1
        else:
            array_to_fill[household_id][1] = 0

    return array_to_fill

