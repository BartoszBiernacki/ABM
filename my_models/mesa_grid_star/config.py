

class Config:
    """
    Class to store multiple settings in one place.
    """
    def __init__(self):
        pass
    
    # voivodeship = 'łódzkie'
    voivodeship = 'lubelskie'
    percent_of_touched_counties = 30
    days_to_fit_death_toll = 60

    # avg_directory = ('results/'
    #                  'Łódzkie/'
    #                  'Runs=11___'
    #                  'Grid_size=(33, 33)___'
    #                  'N=751___'
    #                  'Customers_in_household=3___'
    #                  'Infected_cashiers_at_start=33___'
    #                  'Infect_housemates_boolean=0/'
    #                  'raw data/')
    
    avg_directory = ('results/'
                     'Lubelskie/'
                     'Runs=10___'
                     'Grid_size=(29, 29)___'
                     'N=835___'
                     'Customers_in_household=3___'
                     'Infected_cashiers_at_start=29___'
                     'Infect_housemates_boolean=0/'
                     'raw data/')
    
    not_avg_directory = 'TMP_SAVE/'
