from my_math_utils import *


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


def get_number_of_total_ordinary_pearson_agents(model):
    return model.total_num_of_ordinary_agents


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


@njit
def get_order_of_shopping_in_neighbouring_neighbourhoods(neighbouring_neighbourhoods_grouped_by_neighbourhood_id,
                                                         array_to_fill):
    # result[neighbourhood_id][num_of_specific_household_in_that_neighbourhood] --> list of neighbouring
    # neighbourhoods.
    # Elements in list are in random order.
    z, y, x = np.shape(array_to_fill)
    # z = num of households in neighbourhood
    # y = num of all neighbourhoods
    # x = num of neighbouring neighbourhoods
    for i in range(z):
        for j in range(y):
            array_to_fill[i][j] = np.random.permutation(neighbouring_neighbourhoods_grouped_by_neighbourhood_id[j])
    return array_to_fill


@njit
def get_extra_shopping_days_for_each_household(total_num_of_households, array_to_fill, width, height):
    # [household_id] --> [day_in_week_1, day_in_week_2, ..., day_in_week_n], n={0, 1, 2, 3, 4} depends on grid
    needed_number_of_weeks = int(get_number_of_cell_neighbours(y=height, x=width))
    array_to_fill = array_to_fill.T
    for i in range(0, needed_number_of_weeks*7, 7):
        week = np.random.randint(i, i+7, total_num_of_households)
        array_to_fill[int(i/7)] = week

    result = array_to_fill.T
    return result
