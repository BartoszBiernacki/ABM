from my_math_utils import *


@njit(cache=True)
def get_incubation_period(avg_incubation_period,
                          incubation_period_bins,
                          S_incubation,
                          exponents_incubation):
    
    return int_from_hist(mean=avg_incubation_period,
                         bins=incubation_period_bins,
                         S=S_incubation,
                         exponents=exponents_incubation)


@njit(cache=True)
def get_prodromal_period(avg_prodromal_period,
                         prodromal_period_bins,
                         S_prodromal,
                         exponents_prodromal):
    
    return int_from_hist(mean=avg_prodromal_period,
                         bins=prodromal_period_bins,
                         S=S_prodromal,
                         exponents=exponents_prodromal)


@njit(cache=True)
def get_illness_period(avg_illness_period,
                       illness_period_bins,
                       S_illness,
                       exponents_illness):
    return int_from_hist(mean=avg_illness_period,
                         bins=illness_period_bins,
                         S=S_illness,
                         exponents=exponents_illness)


def find_neighbouring_neighbourhoods(all_cashiers, neighbourhood_pos_to_id):
    # [neighbourhood_id] --> [neighbour_neighbourhood_1_id, ..., neighbour_neighbourhood_n_id], n in {0, 1, 2, 3, 4}
    possible_steps_as_neighbourhood_ids = []
    for i, cashier in enumerate(all_cashiers):
        pos, possible_steps = cashier.show_neighbourhood()
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
def neigh_id_of_extra_shopping(nearest_neighbourhoods_by_neigh_id, neigh_id_by_house_id, array_to_fill):
    # result[week][household_id] --> extra neighbourhood_id to visit
    
    max_weeks, total_households = array_to_fill.shape
    total_neighbourhoods, num_of_nearest_neighbourhoods = nearest_neighbourhoods_by_neigh_id.shape

    if num_of_nearest_neighbourhoods > 0:  # do sth if there is at least one neighbour
    
        covered_cycles = 0
        weeks_to_cover = max_weeks
        
        while True:
            for i in range(min(weeks_to_cover, num_of_nearest_neighbourhoods)):
                for house_id in range(total_households):
                    neigh_id = neigh_id_by_house_id[house_id]
                    shuffled_neighbours = np.random.permutation(nearest_neighbourhoods_by_neigh_id[neigh_id])
                    array_to_fill[covered_cycles * num_of_nearest_neighbourhoods + i][house_id] = shuffled_neighbours[i]
                    
            covered_cycles += 1
            weeks_to_cover -= num_of_nearest_neighbourhoods
        
            if weeks_to_cover <= 0:
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
        else:
            array_to_fill[household_id][1] = 0

    return array_to_fill


# TODO take a look why else ...= False causes problem
@njit(cache=True)
def set_on_shopping_true_depends_on_shopping_situation(total_num_of_households,
                                                       situation,
                                                       array_to_fill):
    for household_id in range(total_num_of_households):
        if situation[household_id][1]:
            volunteer_pos = situation[household_id][0]
            array_to_fill[household_id][volunteer_pos] = True
       
            
@njit(cache=True)
def set_on_extra_shopping_true_depends_on_shopping_situation(total_num_of_households, situation, array_to_fill):
    for household_id in range(total_num_of_households):
        if situation[household_id][1]:
            volunteer_pos = situation[household_id][0]
            array_to_fill[household_id][volunteer_pos] = True


@njit(cache=True)
def find_out_who_wants_to_do_extra_shopping(extra_shopping_days,
                                            day,
                                            agents_state_grouped_by_households,
                                            array_to_fill):

    total_num_of_households, num_of_shopping_days_in_week = array_to_fill.shape

    day_mod_7 = day % 7
    week = day // 7
    # return[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
    for household_id in range(total_num_of_households):
        if day_mod_7 in extra_shopping_days[week][household_id]:
            volunteer_pos = np.argmax(agents_state_grouped_by_households[household_id] <= 2)
            array_to_fill[household_id][0] = volunteer_pos

            if agents_state_grouped_by_households[household_id][volunteer_pos] <= 2:
                array_to_fill[household_id][1] = 1
        else:
            array_to_fill[household_id][1] = 0

    return array_to_fill


@njit(cache=True)
def make_ordinary_shopping_core(total_households,
                                num_of_customers_in_household,
                                A_on_shopping_by_house_id,
                                A_state_by_house_id,
                                C_state_by_neigh_id,
                                neigh_id_by_house_id,
                                beta,
                                A_go_incubation_because_shopping,
                                C_go_incubation_because_shopping):
    customers = 0
    recovery_customers = 0
    susceptible_customers = 0
    incubation_customers = 0
    prodromal_customers = 0
    infected_by_self_cashier = 0
    
    for house_id in range(total_households):
        for h_member in range(num_of_customers_in_household):
            if A_on_shopping_by_house_id[house_id][h_member]:
                customers += 1
                if A_state_by_house_id[house_id][h_member] == -1:
                    recovery_customers += 1
                    A_on_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 0:
                    susceptible_customers += 1
                    if C_state_by_neigh_id[neigh_id_by_house_id[house_id]] == 2:
                        if np.random.rand() <= beta:
                            A_go_incubation_because_shopping[house_id][h_member] = True
                            infected_by_self_cashier += 1
                    A_on_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 1:
                    incubation_customers += 1
                    A_on_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 2:
                    prodromal_customers += 1
                    if C_state_by_neigh_id[neigh_id_by_house_id[house_id]] == 0:
                        if np.random.rand() <= beta:
                            C_go_incubation_because_shopping[neigh_id_by_house_id[house_id]] = True
                    A_on_shopping_by_house_id[house_id][h_member] = False
                    
    if susceptible_customers + incubation_customers + prodromal_customers + recovery_customers != customers:
        raise ValueError("Customers in different stages must add up to customers!")
    return customers, susceptible_customers, incubation_customers, prodromal_customers, recovery_customers, infected_by_self_cashier


@njit(cache=True)
def make_extra_shopping_core(total_households,
                             num_of_customers_in_household,
                             A_on_extra_shopping_by_house_id,
                             A_state_by_house_id,
                             C_state_by_neigh_id,
                             neigh_id_by_house_id,
                             beta,
                             A_go_incubation_because_shopping,
                             C_go_incubation_because_shopping,
                             neigh_id_of_extra_shopping_by_week_and_house_id,
                             day):
    extra_customers = 0
    extra_recovery_customers = 0
    extra_susceptible_customers = 0
    extra_incubation_customers = 0
    extra_prodromal_customers = 0
    infected_by_extra_cashier = 0
    
    for house_id in range(total_households):
        for h_member in range(num_of_customers_in_household):
            if A_on_extra_shopping_by_house_id[house_id][h_member]:
                extra_customers += 1
                if A_state_by_house_id[house_id][h_member] == -1:
                    extra_recovery_customers += 1
                    A_on_extra_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 0:
                    extra_susceptible_customers += 1
                    if C_state_by_neigh_id[neigh_id_by_house_id[house_id]] == 2:
                        if np.random.rand() <= beta:
                            A_go_incubation_because_shopping[house_id][h_member] = True
                            infected_by_extra_cashier += 1
                    A_on_extra_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 1:
                    extra_incubation_customers += 1
                    A_on_extra_shopping_by_house_id[house_id][h_member] = False
                
                elif A_state_by_house_id[house_id][h_member] == 2:
                    extra_prodromal_customers += 1
                    if C_state_by_neigh_id[neigh_id_by_house_id[house_id]] == 0:
                        if np.random.rand() <= beta:
                            C_neigh_id = neigh_id_of_extra_shopping_by_week_and_house_id[day // 7][house_id]
                            C_go_incubation_because_shopping[C_neigh_id] = True
                    A_on_extra_shopping_by_house_id[house_id][h_member] = False

    return extra_customers, extra_susceptible_customers, extra_incubation_customers, extra_prodromal_customers, \
        extra_recovery_customers, infected_by_extra_cashier


@njit(cache=True)
def try_to_infect_housemates_core(total_households,
                                  num_of_customers_in_household,
                                  suspicious_households,
                                  A_state_by_house_id,
                                  A_go_incubation_because_housemate):
    for house_id in range(total_households):
        if suspicious_households[house_id]:
            for house_member in range(num_of_customers_in_household):
                if A_state_by_house_id[house_id][house_member] == 0:
                    A_go_incubation_because_housemate[house_id][house_member] = True
            suspicious_households[house_id] = False


@njit(cache=True)
def update_A_states_core(total_households,
                         num_of_customers_in_household,
                         A_state_by_house_id,
                         suspicious_households,
                         A_go_incubation_because_shopping,
                         A_go_incubation_because_housemate,
                         A_incubation_duration_by_house_id,
                         A_prodromal_duration_by_house_id,
                         A_illness_duration_by_house_id,
                         mortality,
                         prob_of_survive_one_day,
                         die_at_once,
                         avg_incubation_period,
                         incubation_period_bins,
                         S_incubation,
                         exponents_incubation,
                         avg_prodromal_period,
                         prodromal_period_bins,
                         S_prodromal,
                         exponents_prodromal,
                         avg_illness_period,
                         illness_period_bins,
                         S_illness,
                         exponents_illness):
    
    for house_id in range(total_households):
        for house_member in range(num_of_customers_in_household):
            
            if A_state_by_house_id[house_id][house_member] == 0:
                if A_go_incubation_because_shopping[house_id][house_member] or A_go_incubation_because_housemate[house_id][house_member]:
                    A_state_by_house_id[house_id][house_member] = 1
                    A_incubation_duration_by_house_id[house_id][house_member] =\
                        get_incubation_period(avg_incubation_period=avg_incubation_period,
                                              incubation_period_bins=incubation_period_bins,
                                              S_incubation=S_incubation,
                                              exponents_incubation=exponents_incubation)
            
            elif A_state_by_house_id[house_id][house_member] == 1:
                A_incubation_duration_by_house_id[house_id][house_member] -= 1
                if A_incubation_duration_by_house_id[house_id][house_member] <= 0:
                    A_state_by_house_id[house_id][house_member] = 2
                    A_prodromal_duration_by_house_id[house_id][house_member] = \
                        get_prodromal_period(avg_prodromal_period=avg_prodromal_period,
                                             prodromal_period_bins=prodromal_period_bins,
                                             S_prodromal=S_prodromal,
                                             exponents_prodromal=exponents_prodromal)
                    
                    if num_of_customers_in_household > 1:
                        suspicious_households[house_id] = True
            
            elif A_state_by_house_id[house_id][house_member] == 2:
                A_prodromal_duration_by_house_id[house_id][house_member] -= 1
                if A_prodromal_duration_by_house_id[house_id][house_member] <= 0:
                    A_state_by_house_id[house_id][house_member] = 3
                    A_illness_duration_by_house_id[house_id][house_member] = \
                        get_illness_period(avg_illness_period=avg_illness_period,
                                           illness_period_bins=illness_period_bins,
                                           S_illness=S_illness,
                                           exponents_illness=exponents_illness)
            
            elif A_state_by_house_id[house_id][house_member] == 3:
                A_illness_duration_by_house_id[house_id][house_member] -= 1
                
                if die_at_once:
                    if A_illness_duration_by_house_id[house_id][house_member] <= 0:
                        if np.random.rand() <= mortality:
                            A_state_by_house_id[house_id][house_member] = 4
                        else:
                            A_state_by_house_id[house_id][house_member] = -1
                else:
                    if np.random.rand() > prob_of_survive_one_day:  # if die today:
                        A_illness_duration_by_house_id[house_id][house_member] = 0
                        A_state_by_house_id[house_id][house_member] = 4
                        
                    elif A_illness_duration_by_house_id[house_id][house_member] <= 0:
                        A_state_by_house_id[house_id][house_member] = -1
                
    A_go_incubation_because_shopping.fill(False)
    A_go_incubation_because_housemate.fill(False)
    
                        
@njit(cache=True)
def update_C_states_core(total_neighbourhoods,
                         C_state_by_neigh_id,
                         C_go_incubation_because_shopping,
                         C_incubation_duration_by_neigh_id,
                         C_prodromal_duration_by_neigh_id,
                         avg_incubation_period,
                         incubation_period_bins,
                         S_incubation,
                         exponents_incubation,
                         avg_prodromal_period,
                         prodromal_period_bins,
                         S_prodromal,
                         exponents_prodromal):
    
    replaced_cashiers_today = 0
    for neigh_id in range(total_neighbourhoods):
        if C_state_by_neigh_id[neigh_id] == 0:
            if C_go_incubation_because_shopping[neigh_id]:
                C_state_by_neigh_id[neigh_id] = 1
                C_incubation_duration_by_neigh_id[neigh_id] = \
                    get_incubation_period(avg_incubation_period=avg_incubation_period,
                                          incubation_period_bins=incubation_period_bins,
                                          S_incubation=S_incubation,
                                          exponents_incubation=exponents_incubation)
        
        elif C_state_by_neigh_id[neigh_id] == 1:
            C_incubation_duration_by_neigh_id[neigh_id] -= 1
            if C_incubation_duration_by_neigh_id[neigh_id] <= 0:
                C_state_by_neigh_id[neigh_id] = 2
                C_prodromal_duration_by_neigh_id[neigh_id] = \
                    get_prodromal_period(avg_prodromal_period=avg_prodromal_period,
                                         prodromal_period_bins=prodromal_period_bins,
                                         S_prodromal=S_prodromal,
                                         exponents_prodromal=exponents_prodromal)
        
        elif C_state_by_neigh_id[neigh_id] == 2:
            C_prodromal_duration_by_neigh_id[neigh_id] -= 1
            if C_prodromal_duration_by_neigh_id[neigh_id] <= 0:
                C_state_by_neigh_id[neigh_id] = 0
                replaced_cashiers_today += 1
                
    C_go_incubation_because_shopping.fill(False)
    return replaced_cashiers_today
    
    
