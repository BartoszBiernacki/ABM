from my_math_utils import *


# POPULATION STATE RELATED ============================================================================================
# Ordinary agents---------------------------------------------------
def calculate_ordinary_pearson_number_susceptible(model):
    return int(np.sum(model.A_state_by_house_id == 0))


def calculate_ordinary_pearson_number_incubation(model):
    return int(np.sum(model.A_state_by_house_id == 1))


def calculate_ordinary_pearson_number_prodromal(model):
    return int(np.sum(model.A_state_by_house_id == 2))


def calculate_ordinary_pearson_number_illness(model):
    return int(np.sum(model.A_state_by_house_id == 3))


def calculate_ordinary_pearson_number_dead(model):
    return int(np.sum(model.A_state_by_house_id == 4))


def calculate_ordinary_pearson_number_recovery(model):
    return int(np.sum(model.A_state_by_house_id == -1))


# Cashiers ---------------------------------------------------
def calculate_susceptible_cashiers(model):
    return int(np.sum(model.C_state_by_neigh_id == 0))


def calculate_incubation_cashiers(model):
    return int(np.sum(model.C_state_by_neigh_id == 1))


def calculate_prodromal_cashiers(model):
    return int(np.sum(model.C_state_by_neigh_id == 2))


def calculate_replaced_cashiers(model):
    return model.replaced_cashiers
# *********************************************************************************************************************


# ORDINARY SHOPPING RELATED ===========================================================================================
def calculate_ordinary_customers(model):
    return model.ordinary_customers


def calculate_susceptible_customers(model):
    return model.susceptible_customers


def calculate_incubation_customers(model):
    return model.incubation_customers


def calculate_prodromal_customers(model):
    return model.prodromal_customers


def calculate_recovery_customers(model):
    return model.recovery_customers


def calculate_infected_by_self_cashier_today(model):
    return model.infected_customers_by_self_cashier
# *********************************************************************************************************************


# EXTRA SHOPPING RELATED ==============================================================================================
def calculate_extra_customers(model):
    return model.extra_customers


def calculate_extra_susceptible_customers(model):
    return model.extra_susceptible_customers


def calculate_extra_incubation_customers(model):
    return model.extra_incubation_customers


def calculate_extra_prodromal_customers(model):
    return model.extra_prodromal_customers


def calculate_extra_recovery_customers(model):
    return model.extra_recovery_customers


def calculate_infected_by_extra_cashier_today(model):
    return model.infected_customers_by_extra_cashier
# *********************************************************************************************************************


# OTHERS =============================================================================================================
def get_day(model):
    return model.day


def calculate_execution_time(model):
    return model.execution_time
# *********************************************************************************************************************
