import os


class Config:
    """
    Class to store multiple settings in one place.
    """
    def __init__(self):
        pass
    
    ABM_dir = os.path.split(
        os.path.dirname(os.path.realpath(__file__)))[0].replace("\\", "/")

    TMP_SAVE_DIR = (f"{ABM_dir}/"
                    f"TMP_SAVE/")

    AVG_SAVE_DIR = (f"{ABM_dir}/"
                    f"RESULTS/")
    
    days_to_look_for_pandemic_end = 200
    death_toll_derivative_half_win_size = 3
    death_toll_derivative_smooth_out_win_size = 21
    death_toll_derivative_smooth_out_savgol_polyorder = 3
    
    days_to_fit_death_toll = 60
    customers_in_household = 3
    
    percent_of_touched_counties = 30
    voivodeship = 'łódzkie'

    avg_directory = (
        f'{AVG_SAVE_DIR}'
        f'{voivodeship.capitalize()}/'
        'Runs=11___'
        'Grid_size=(33, 33)___'
        'N=751___'
        'Customers_in_household=3___'
        'Infected_cashiers_at_start=33___'
        'Infect_housemates_boolean=0/'
        'raw data/')
    
    not_avg_directory = 'TMP_SAVE/'
