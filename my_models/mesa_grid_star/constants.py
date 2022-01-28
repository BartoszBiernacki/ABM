from collections import OrderedDict
import pandas as pd


def get_population_density():
    population_density_mixed = {'dolnośląskie': 145,
                                'kujawsko-pomorskie': 115,
                                'lubelskie': 83,
                                'lubuskie': 72,
                                'łódzkie': 134,
                                'małopolskie': 225,
                                'mazowieckie': 153,
                                'opolskie': 104,
                                'podkarpackie': 119,
                                'podlaskie': 58,
                                'pomorskie': 128,
                                'śląskie': 364,
                                'świętokrzyskie': 105,
                                'warmińsko-mazurskie': 59,
                                'wielkopolskie': 117,
                                'zachodniopomorskie': 74}
    
    population_density = OrderedDict(sorted(population_density_mixed.items(),
                                            key=lambda kv: kv[1]))
    return population_density


def get_urbanization():
    df = pd.read_csv('real_data/general/GUS_general_pop_data.csv')
    density_2019 = df['Unnamed: 6'].to_list()
    
    for i, val in enumerate(density_2019):
        val = val.replace(',', '.')
        density_2019[i] = val
    df['Unnamed: 6'] = density_2019
    
    to_care = ['Województwa/ lata', 'Unnamed: 6']
    new_df = df[to_care]
    
    new_df = new_df.drop([0, 1])
    new_df.columns = ['województwo', 'wsp_urbanizacji']
    new_df.set_index('województwo')
    
    urbanization_mixed = {row[0].lower(): row[1] for row in new_df.values}
    urbanization = OrderedDict(sorted(urbanization_mixed.items(),
                                      key=lambda kv: kv[1]))
    return urbanization


fits = {
    'podlaskkie': (0.022, ((210, 420), (1.95, 0.6)), 2, 0.65),
    'lubuskie': (0.028, ((100, 158, 330), (1.8, 0.6, 10)), 2, 0.65),
    'opolskie': (0.040, ((20, 70, 120, 155), (0.5, 0.8, 1.4, 0.3)), 2, 0.65),
    
}
