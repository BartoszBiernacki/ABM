fname = (
    '../'
    'data/'
    'raw/'
    'geospatial/'
    'GUS_general_pop_data.csv')


import pandas as pd
import os
from disease_spread_model.config import Config

abm = Config.ABM_dir
abm = abm.replace("\\", "/")
print(abm)