import os
from enum import Enum, auto


class StartingDayBy(Enum):
    INFECTIONS = auto()
    DEATHS = auto()


class ModelOptions:
    """Holds 'DiseaseModel' default parameters."""
    
    GRID_SIZE = (20, 20)
    N = 700
    CUSTOMERS_IN_HOUSEHOLD = 3
    BETA = 2.5/100
    MORTALITY = 2.0/100
    VISIBILITY = 65/100
    
    AVG_INCUBATION_PERIOD, INCUBATION_PERIOD_BINS = 5, 3
    AVG_PRODROMAL_PERIOD, PRODROMAL_PERIOD_BINS = 3, 3
    AVG_ILLNESS_PERIOD, ILLNESS_PERIOD_BINS = 15, 1

    INFECTED_CASHIERS_AT_START = 20
    # PERCENT_OF_INFECTED_CUSTOMERS_AT_START = 2
    PERCENT_OF_INFECTED_CUSTOMERS_AT_START = 0
    HOUSEMATE_INFECTION_PROBABILITY = 0.53
    EXTRA_SHOPPING_BOOLEAN = True
    MAX_STEPS = 200

    DEFAULT_MODEL_PARAMS = {
        'grid_size': GRID_SIZE,
        'N': N,
        'customers_in_household': CUSTOMERS_IN_HOUSEHOLD,
        'beta': BETA,
        'mortality': MORTALITY,
        'visibility': VISIBILITY,
    
        'avg_incubation_period': AVG_INCUBATION_PERIOD,
        'incubation_period_bins': INCUBATION_PERIOD_BINS,
        'avg_prodromal_period': AVG_PRODROMAL_PERIOD,
        'prodromal_period_bins': PRODROMAL_PERIOD_BINS,
        'avg_illness_period': AVG_ILLNESS_PERIOD,
        'illness_period_bins': ILLNESS_PERIOD_BINS,
    
        'infected_cashiers_at_start': INFECTED_CASHIERS_AT_START,
        'percent_of_infected_customers_at_start': PERCENT_OF_INFECTED_CUSTOMERS_AT_START,
        'extra_shopping_boolean': EXTRA_SHOPPING_BOOLEAN,
        'housemate_infection_probability': HOUSEMATE_INFECTION_PROBABILITY,
        'max_steps': MAX_STEPS,
    }


class Directories:
    """Holds useful paths."""
    
    ABM_DIR = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0].replace("\\", "/")
    
    TMP_SAVE_DIR = f"{ABM_DIR}/TMP_SAVE/"
    AVG_SAVE_DIR = f"{ABM_DIR}/RESULTS/"
    
    NOT_AVG_DIRECTORY = 'TMP_SAVE/'
    
    TUNING_MODEL_PARAMS_FNAME = (
        f"{ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"model_params_tuning/"
        f"model_params_tuning_attempts.csv")
    
    TEST_PLOT_DIR = f"{ABM_DIR}/RESULTS/TESTS/plots/"


class RealDataOptions:
    """Holds some parameters used for finding last and first day of pandemic."""
    
    DAYS_TO_LOOK_FOR_PANDEMIC_END = 200
    PERCENT_OF_DEATH_COUNTIES = 30
    PERCENT_OF_INFECTED_COUNTIES = 80

    DEATH_TOLL_DERIVATIVE_HALF_WIN_SIZE = 7
    DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_WIN_SIZE = 21
    DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_SAVGOL_POLYORDER = 3
    
    STARTING_DAYS_BY = StartingDayBy.INFECTIONS
    LAST_DATE = '2020-07-01'


class PlotOptions:
    """Holds some parameters used for plotting."""
    
    
    
    voivodeship = 'łódzkie'
    
    # avg_directory = (
    #     f'{AVG_SAVE_DIR}'
    #     f'{voivodeship.capitalize()}/'
    #     'Runs=11___'
    #     'Grid_size=(33, 33)___'
    #     'N=751___'
    #     'Customers_in_household=3___'
    #     'Infected_cashiers_at_start=33___'
    #     'Infect_housemates_boolean=0/'
    #     'raw data/')


if __name__ == '__main__':
    print(ModelOptions.HOUSEMATE_INFECTION_PROBABILITY)