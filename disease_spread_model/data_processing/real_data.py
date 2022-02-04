"""Extracting real pandemic and geospatial data"""
import numpy as np
import pandas as pd
import codecs
import pickle

from disease_spread_model.data_processing.text_processing import all_fnames_from_dir
from disease_spread_model.data_processing.text_processing import rename_duplicates_in_df_index_column
from disease_spread_model.config import Config


class RealData(object):
    """
    Purpose of this class is to provide following real data:
        * get_voivodeships() - list of voivodeships
        * get_real_general_data() - df with geographic data (unrelated to pandemic) for voivodeships
        * get_real_death_toll() - df with death toll dynamic for voivodeships
        * get_real_infected_toll() - df with infected toll dynamic for voivodeships
        * get_day_of_first_n_death() - dict {voivodeship: day}
        * get_starting_days_for_voivodeships_based_on_district_deaths() - dict {voivodeship: day}, day is an earliest
            day in which P percent of all counties in voivodeship a death case has been reported.
        * get_starting_deaths_by_hand() - dict {voivodeship: starting_death_toll}, starting_death_toll is a minimum
            number of deaths since which death tool curve looks nicely to me.
        * get_starting_death_toll_for_voivodeships_by_days() - {voivodeship: starting_death_toll}, death toll in
            voivodeships in days specified by external dict like {voivodeship: day}
        * get_shifted_real_death_toll_to_common_start_by_num_of_deaths() - df where at given day in all voivodeships
            there are at least n deaths
    """
    __voivodeships = ['dolnośląskie',
                      'kujawsko-pomorskie',
                      'lubelskie',
                      'lubuskie',
                      'łódzkie',
                      'małopolskie',
                      'mazowieckie',
                      'opolskie',
                      'podkarpackie',
                      'podlaskie',
                      'pomorskie',
                      'śląskie',
                      'świętokrzyskie',
                      'warmińsko-mazurskie',
                      'wielkopolskie',
                      'zachodniopomorskie'
                      ]
    
    # http://eregion.wzp.pl/obszary/stan-i-struktura-ludnosci
    __fname_GUS_general_pop_data = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"geospatial/"
        f"GUS_general_pop_data.csv")

    __fname_counties_in_voivodeship_final = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"geospatial/"
        f"counties_in_voivodeship.pck")
    
    # http://eregion.wzp.pl/liczba-mieszkancow-przypadajacych-na-1-sklep
    __fname_peoples_per_shop = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"geospatial/"
        f"mieszkancow_na_sklep.csv")
    
    # POPULATION DENSITY DATA SOURCE (entered manually)
    # https://stat.gov.pl/obszary-tematyczne/ludnosc/
    # ludnosc/powierzchnia-i-ludnosc-w-przekroju-terytorialnym-w-2021-roku,7,18.html
    
    # https://bit.ly/covid19_powiaty
    __fname_pandemic_day_by_day_early = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"pandemic/"
        f"early_data/"
        f"COVID-19_04.03-23.11.xlsm")

    __dir_pandemic_day_by_day_late_raw = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"pandemic/"
        f"late_data/")

    __fname_pandemic_day_by_day_late_final = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"pandemic_day_by_day.pck")
    
    __fname_entire_death_toll_final = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"entire_death_toll.pck")
    
    __fname_entire_infected_toll_final = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"entire_infected_toll.pck")
    
    __fname_df_excel_deaths_pandemic_early_final = (
        f"{Config.ABM_dir}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"df_excel_deaths_pandemic_early_final.pck")
    
    def __init__(self):
        pass
    
    # Get voivodeships ************************************************************************************************
    @classmethod
    def get_voivodeships(cls):
        """ Returns list containing lowercase voivodeship names. """
        return cls.__voivodeships
    
    # Get counties in voivodeship ************************************************************************************
    @classmethod
    def __get_counties_in_voivodeship(cls):
        """
        Returns dict in which key = voivodeship, value = list of counties in it.
        """
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma przypadków'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        df_excel.drop(columns=['Kod', "Unnamed: 1"], inplace=True)
        df_excel.drop([0, 1], inplace=True)
        
        voivodeships = cls.get_voivodeships()
        counties_in_voivodeship = {}
        counties = []
        voivodeship = None
        for name in df_excel['Nazwa']:
            if pd.notna(name):
                if name.lower() in voivodeships:
                    voivodeship = name
                    counties = []
                else:
                    counties.append(name)
            else:
                counties_in_voivodeship[voivodeship] = counties
        
        # lowercase voivodeships to be consistent in general
        counties_in_voivodeship = {k.lower(): v for k, v in counties_in_voivodeship.items()}
        return counties_in_voivodeship
    
    @classmethod
    def __save_counties_in_voivodeship_as_pickle(cls):
        counties_in_voivodeship = cls.__get_counties_in_voivodeship()
        
        with open(cls.__fname_counties_in_voivodeship_final, 'wb') as handle:
            pickle.dump(counties_in_voivodeship, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def __load_counties_in_voivodeship_from_pickle(cls):
        with open(cls.__fname_counties_in_voivodeship_final, 'rb') as handle:
            counties_in_voivodeship = pickle.load(handle)
        return counties_in_voivodeship
    
    @classmethod
    def get_counties_in_voivodeship(cls):
        """
        Returns dict in which: key = voivodeship name, value = list of counties in it.
        """
        try:
            counties_in_voivodeship = cls.__load_counties_in_voivodeship_from_pickle()
        except FileNotFoundError:
            cls.__save_counties_in_voivodeship_as_pickle()
            counties_in_voivodeship = cls.__load_counties_in_voivodeship_from_pickle()
        
        return counties_in_voivodeship
    
    # Get data about population, population density, urbanization and shops among voivodeships [2019 and 2021] *******
    @classmethod
    def get_real_general_data(cls, customers_in_household=3):
        """
        Get DataFrame with real population data independent of the pandemic like
        urbanization , population density and number of shops.
        DataFrame will also include columns with some recommended model parameters for each voivodeship like:
        grid side or N - number of households in one grid cell.

        Note 1: recommended model parameters are in columns named ,,xxx MODEL''.
        Note 2: ,,customers_in_household'' only affects on recommended number of households in one grid cell i.e.
            column ,,N MODEL''

        Columns = ['population', 'urbanization', 'population density', 'peoples per shop', 'shops', 'shops MODEL',
                    'grid side MODEL', 'N MODEL']
        """
        
        df = pd.read_csv(cls.__fname_GUS_general_pop_data)
        df = df.drop([0, 1])  # drop sub column label and data for entire country
        
        # manually set column names, because they are broken
        df.columns = ['voivodeship',
                      'pop cities 2010', 'pop cities 2019',
                      'pop village 2010', 'pop village 2019',
                      'urbanization 2010', 'urbanization 2019']
        
        # drop not interesting columns
        df.drop(columns=['pop cities 2010', 'pop village 2010', 'urbanization 2010'], inplace=True)
        
        # make column values compatible with the accepted convention
        df['voivodeship'] = df['voivodeship'].str.lower()
        
        # Convert strings to numbers
        df['pop cities 2019'] = [int(item.replace(' ', '')) for item in df['pop cities 2019']]
        df['pop village 2019'] = [int(item.replace(' ', '')) for item in df['pop village 2019']]
        df['urbanization 2019'] = [float(item.replace(',', '.')) for item in df['urbanization 2019']]
        
        # Make new column having population of voivodeships
        df['population'] = df['pop cities 2019'] + df['pop village 2019']
        
        # drop not interesting columns
        df.drop(columns=['pop cities 2019', 'pop village 2019'], inplace=True)
        
        # set new names to columns as now there are only data from 2019 (not 2010)
        df.columns = ['voivodeship', 'urbanization', 'population']
        
        # set voivodeship column as an index column
        df.set_index('voivodeship', inplace=True)
        
        # explicitly leave the columns chosen
        df = df[['population', 'urbanization']]
        # ----------------------------------------------------------------------------------------------------------
        
        # Get data about population density from GUS webpage [2021] --------------------------------------------------
        # https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5468/7/18/1/
        # powierzchnia_i_ludnosc_w_przekroju_terytorialnym_w_2021_roku_tablice.xlsx
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
        
        # add new column with population density
        df['population density'] = [val for val in population_density_mixed.values()]
        # ----------------------------------------------------------------------------------------------------------
        
        # Get data about num of peoples per shop [2021 to new temporary DataFrame ---------------------------
        tmp_df = pd.read_csv(cls.__fname_peoples_per_shop)
        tmp_df.drop(0, inplace=True)
        tmp_df.rename(columns={'Województwa/Lata': 'voivodeship'}, inplace=True)
        tmp_df['voivodeship'] = tmp_df['voivodeship'].str.lower()
        tmp_df.set_index('voivodeship', inplace=True)
        
        # Get Series containing data about number of peoples per shop for all voivodeships
        shop_series = pd.Series(tmp_df['2019'], name='peoples per shop')
        
        # Merge previous DataFrame with peoples per shop Series
        df = pd.concat([df, shop_series], axis=1)
        # ----------------------------------------------------------------------------------------------------------
        
        # determine N and grid size based on population and peoples per shop ---------------------------------------
        
        # calculate number of shops in reality and rescale it
        shops = df['population'] / df['peoples per shop']
        shops_model = shops / 20
        
        # add data about number of shops to DataFrame
        df['shops'] = shops.astype(int)
        df['shops MODEL'] = shops_model.astype(int)
        
        # add column with grid side length such that: grid_side_length**2 = rescaled num of shops in voivodeship
        grid_side = np.sqrt(df['shops MODEL'])
        df['grid side MODEL'] = grid_side.astype(int)
        
        N_model = df['population'] / customers_in_household
        N_model /= df['grid side MODEL'] ** 2
        df['N MODEL'] = N_model.astype(int)
        # ----------------------------------------------------------------------------------------------------------
        
        return df
    
    # Get data about dead toll and recovered toll among voivodeships during pandemic ******************************
    @classmethod
    def __convert_files_to_UTF8(cls):
        """
            For all files in directory given by self.dir_pandemic_day_by_day_late_raw:
                * converts text files format from ANSI to UTF-8.
                * if original file format is UTF-8 then leaves it as it is.

            Directory self.dir_pandemic_day_by_day_late_raw should contain only
            such text files.

            Python by default support UTF-8 encoding, but this class operates on
            external data files and some of them are in ANSI and others in UTF-8.
        """
        fnames = all_fnames_from_dir(
            directory=cls.__dir_pandemic_day_by_day_late_raw)
        for fname in fnames:
            try:
                with codecs.open(fname, 'r', encoding='UTF-8') as file:
                    file.read()
            except UnicodeDecodeError:
                # read input file
                with codecs.open(fname, 'r', encoding='mbcs') as file:
                    lines = file.read()
                
                # write output file
                with codecs.open(fname, 'w', encoding='utf8') as file:
                    file.write(lines)
                
                print(f"File {fname} converted to UTF-8 from ANSI")
    
    @classmethod
    def __get_significant_pandemic_late_data(cls, fname):
        """
            Extracts data that I care of from one data file shared by GUS.

            This function returns Dataframe where row index are voivodeship names (in lowercase)
            and columns are: ['day', 'liczba_przypadkow', 'liczba_ozdrowiencow', 'zgony']

            One file has data about all voivodeship and one day of pandemic.
        """
        
        # read data
        df = pd.read_csv(fname, sep=';')
        
        # get only filename (no entire directory) to construct date from it
        fname_only = fname
        while '/' in fname_only:
            pos = fname_only.find('/')
            fname_only = fname_only[pos + 1:]
        
        # read day from filename
        day = fname_only[:4] + '-' + fname_only[4:6] + '-' + fname_only[6:8]
        
        # make dataframe about voivodeships not counties
        df = df.groupby(['wojewodztwo']).sum()
        
        # insert day into DataFrame column
        df['day'] = day
        
        # make sure that DataFrame contains ,,recovered people'' column
        if 'liczba_ozdrowiencow' not in df.columns:
            df['liczba_ozdrowiencow'] = np.NaN
        
        # keep only those columns which have data important dor the project
        to_care = ['day', 'liczba_przypadkow', 'liczba_ozdrowiencow', 'zgony']
        df = df[to_care]
        
        # sort dataframe by voivodeship in ascending order
        df.sort_values(by=['wojewodztwo'], inplace=True)
        
        return df
    
    @classmethod
    def __prepare_real_pandemic_late_data(cls):
        """
            Extracts data that I care of from all data files shared by GUS.

            This function returns a dict in which keys are voivodeship names (in lowercase)
            and values are dataframes created by: self.__get_significant_pandemic_late_data
            so columns are ['day', 'liczba_przypadkow', 'liczba_ozdrowiencow', 'zgony']

            Directory that should contains all data files about pandemic from GUS is stored by the
            ,,self.dir_pandemic_day_by_day_late_raw'' variable.
            Directory can't contain any other files.
        """
        
        # make sure that files are readable by default settings
        cls.__convert_files_to_UTF8()
        
        # get list of files from given directory
        fnames = all_fnames_from_dir(directory=cls.__dir_pandemic_day_by_day_late_raw)
        
        # prepare dict which will be returned by this method
        result_dict = {}
        
        # create empty Dataframes having a desired format (same like read dataframe ,,df'')
        df = cls.__get_significant_pandemic_late_data(fname=fnames[0])
        voivodeships = df.index.to_list()
        cols = df.columns.to_list()
        for voivodeship in voivodeships:
            result_dict[voivodeship] = pd.DataFrame(columns=cols)
        
        # one day of pandemic is one file so iterate over them, grab data and insert as rows to Dataframes
        for fname in fnames:
            df = cls.__get_significant_pandemic_late_data(fname=fname)
            # modify result voivodeships adding read data, next file = new row
            for voivodeship in voivodeships:
                voivodeship_df = result_dict[voivodeship]
                
                voivodeship_df.loc[-1] = df.loc[voivodeship, :].values  # adding a row
                voivodeship_df.index = voivodeship_df.index + 1  # shifting index
                voivodeship_df.sort_index()  # sorting by index
        
        # Sort pandemic DataFrames row by days (chronological)
        for val in result_dict.values():
            val.sort_values(by=['day'], inplace=True)
            val.reset_index(drop=True, inplace=True)
        
        return result_dict
    
    @classmethod
    def __save_real_late_data_as_pickle(cls):
        """
        Saves data obtained by function ,,__prepare_real_pandemic_late_data''
        to a binary file.

        This function is for time saving, because obtaining data from data files
        given by GUS is time consuming and I may need to do get them over and over again.

        Function saves obtained data to file given by
        ,,self.fname_pandemic_day_by_day_late_final'' variable
        """
        real_late_data = cls.__prepare_real_pandemic_late_data()
        
        with open(cls.__fname_pandemic_day_by_day_late_final, 'wb') as handle:
            pickle.dump(real_late_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def __load_real_data_from_pickle(cls):
        """
        Loads data previously saved by function ,,self.__save_real_late_data_as_pickle''.

        This function is for time saving, because obtaining data from data files
        given by GUS is time consuming and I may need to do get them over and over again.

        Function loads obtained data from file given by
        ,,self.fname_pandemic_day_by_day_late_final'' variable
        """
        with open(cls.__fname_pandemic_day_by_day_late_final, 'rb') as handle:
            real_data = pickle.load(handle)
        return real_data
    
    @classmethod
    def __get_real_late_pandemic_data(cls):
        """
        Returns data obtained by function ,,self.__prepare_real_pandemic_late_data'',
        but instead of calling it every time, calls it once and then save data to binary file and
        when called reads data from that file.

        If file stored in directory ,,self.fname_pandemic_day_by_day_late_final'' exist then it
        is considered as function ,,self.__prepare_real_pandemic_late_data'' was already called
        and result of it is stored in dir given by ,,self.fname_pandemic_day_by_day_late_final'' variable.
        """
        try:
            real_pandemic_data = cls.__load_real_data_from_pickle()
        except FileNotFoundError:
            cls.__save_real_late_data_as_pickle()
            real_pandemic_data = cls.__load_real_data_from_pickle()
        
        return real_pandemic_data
    
    @classmethod
    def __get_death_toll_at_end_of_early_pandemic(cls):
        """
        Returns death toll at the first day when GUS started to publish it's data.

        This function helps merge data from GUS and from private dataset.
        First source (private source) has death toll, but second (GUS) has deaths in each day.

        Finally I want to have death toll at the beginning of pandemic to current day.

        Directory to the file that contains data from private source is stored in
        ,,self.fname_pandemic_day_by_day_early'' variable
        """
        
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma zgonów'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        
        valid_rows = [voivodeship.upper() for voivodeship in cls.get_voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=[158, 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)
        
        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)
        
        death_toll_initial = df_excel.max(axis=1)
        return death_toll_initial
    
    @classmethod
    def __get_death_toll_for_early_pandemic(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma zgonów'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        
        valid_rows = [voivodeship.upper() for voivodeship in cls.get_voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=[158, 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)
        
        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)
        
        return df_excel
    
    @classmethod
    def __merge_properly_early_and_late_pandemic_death_toll(cls):
        late_pandemic_data = cls.__get_real_late_pandemic_data()
        
        late_days = late_pandemic_data['Cały kraj']['day'].to_list()
        late_pandemic_death_toll = pd.DataFrame(columns=['voivodeship'] + late_days)
        late_pandemic_death_toll.set_index('voivodeship', inplace=True)
        
        death_toll_at_end_of_early_stage = cls.__get_death_toll_at_end_of_early_pandemic()
        for voivodeship, df in late_pandemic_data.items():
            if voivodeship != 'Cały kraj':
                late_pandemic_death_toll.loc[voivodeship] = \
                    np.cumsum(df['zgony'].to_list()) + death_toll_at_end_of_early_stage[voivodeship]
        
        late_pandemic_death_toll = late_pandemic_death_toll.astype(int)
        
        early_pandemic_death_toll = cls.__get_death_toll_for_early_pandemic()
        
        pandemic_death_toll = early_pandemic_death_toll.merge(late_pandemic_death_toll,
                                                              on='voivodeship', how='inner')
        
        return pandemic_death_toll
    
    @classmethod
    def __save_entire_death_toll_as_pickle(cls):
        entire_death_toll = cls.__merge_properly_early_and_late_pandemic_death_toll()
        
        with open(cls.__fname_entire_death_toll_final, 'wb') as handle:
            pickle.dump(entire_death_toll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def __load_entire_death_toll_from_pickle(cls):
        with open(cls.__fname_entire_death_toll_final, 'rb') as handle:
            entire_death_toll = pickle.load(handle)
        return entire_death_toll
    
    @classmethod
    def get_real_death_toll(cls):
        try:
            entire_death_toll = cls.__load_entire_death_toll_from_pickle()
        except FileNotFoundError:
            cls.__save_entire_death_toll_as_pickle()
            entire_death_toll = cls.__load_entire_death_toll_from_pickle()
        
        return entire_death_toll
    
    # Get data about infected toll and recovered toll among voivodeships during pandemic **************************
    @classmethod
    def __get_infected_toll_for_early_pandemic(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma przypadków'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        
        valid_rows = [voivodeship.upper() for voivodeship in cls.get_voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=['Kod', 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)
        
        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)
        
        return df_excel
    
    @classmethod
    def __get_infected_toll_at_end_of_early_pandemic(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma przypadków'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        
        valid_rows = [voivodeship.upper() for voivodeship in cls.get_voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=['Kod', 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)
        
        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)
        
        infected_toll_initial = df_excel.max(axis=1)
        
        return infected_toll_initial
    
    @classmethod
    def __merge_properly_early_and_late_pandemic_infected_toll(cls):
        late_pandemic_data = cls.__get_real_late_pandemic_data()
        
        late_days = late_pandemic_data['Cały kraj']['day'].to_list()
        late_pandemic_infected_toll = pd.DataFrame(columns=['voivodeship'] + late_days)
        late_pandemic_infected_toll.set_index('voivodeship', inplace=True)
        
        infected_toll_at_end_of_early_stage = cls.__get_infected_toll_at_end_of_early_pandemic()
        for voivodeship, df in late_pandemic_data.items():
            if voivodeship != 'Cały kraj':
                late_pandemic_infected_toll.loc[voivodeship] = \
                    np.cumsum(df['liczba_przypadkow'].to_list()) + infected_toll_at_end_of_early_stage[voivodeship]
        
        late_pandemic_infected_toll = late_pandemic_infected_toll.astype(int)
        
        early_pandemic_infected_toll = cls.__get_infected_toll_for_early_pandemic()
        
        pandemic_infected_toll = early_pandemic_infected_toll.merge(late_pandemic_infected_toll,
                                                                    on='voivodeship', how='inner')
        
        return pandemic_infected_toll
    
    @classmethod
    def __save_entire_infected_toll_as_pickle(cls):
        entire_infected_toll = cls.__merge_properly_early_and_late_pandemic_infected_toll()
        with open(cls.__fname_entire_infected_toll_final, 'wb') as handle:
            pickle.dump(entire_infected_toll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def __load_entire_infected_toll_from_pickle(cls):
        with open(cls.__fname_entire_infected_toll_final, 'rb') as handle:
            entire_infected_toll = pickle.load(handle)
        return entire_infected_toll
    
    @classmethod
    def get_real_infected_toll(cls):
        try:
            entire_infected_toll = cls.__load_entire_infected_toll_from_pickle()
        except FileNotFoundError:
            cls.__save_entire_infected_toll_as_pickle()
            entire_infected_toll = cls.__load_entire_infected_toll_from_pickle()
        
        return entire_infected_toll
    
    # Get data about pandemic progress in it's early stage (before data from GUS)
    @classmethod
    def __get_df_excel_deaths_pandemic_early(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma zgonów'
        
        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)
        df_excel.drop(columns=[158, "Unnamed: 1"], inplace=True)
        df_excel.drop([0, 1], inplace=True)
        df_excel.set_index('Nazwa', inplace=True)
        
        # some counties have the same name which causes problem so I'll rename them ...
        df_excel = rename_duplicates_in_df_index_column(df_excel)
        
        return df_excel
    
    @classmethod
    def __save_df_excel_deaths_pandemic_early_as_pickle(cls):
        df_excel_deaths_pandemic_early = cls.__get_df_excel_deaths_pandemic_early()
        
        with open(cls.__fname_df_excel_deaths_pandemic_early_final, 'wb') as handle:
            pickle.dump(df_excel_deaths_pandemic_early, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def __load_df_excel_deaths_pandemic_early_from_pickle(cls):
        with open(cls.__fname_df_excel_deaths_pandemic_early_final, 'rb') as handle:
            df_excel_deaths_pandemic_early = pickle.load(handle)
        return df_excel_deaths_pandemic_early
    
    @classmethod
    def __get_df_excel_deaths_early(cls):
        """
        Returns data from excel file about deaths in early stage of pandemic as
        pandas DataFrame
        """
        try:
            df_excel_deaths_pandemic_early = cls.__load_df_excel_deaths_pandemic_early_from_pickle()
        except FileNotFoundError:
            cls.__save_df_excel_deaths_pandemic_early_as_pickle()
            df_excel_deaths_pandemic_early = cls.__load_df_excel_deaths_pandemic_early_from_pickle()
        
        return df_excel_deaths_pandemic_early
    
    # Get first day of pandemic based on number of deaths ************************************************************
    @classmethod
    def get_day_of_first_n_death(cls, n):
        """
        This function returns dict in which keys are voivodeship names (lowercase) and
        value is a day (YYYY-MM-DD) of first death in given voivodeship.

        Useful function to define beginning of pandemic in given voivodeship.
        """
        real_death_toll = cls.get_real_death_toll()
        first_death = {}
        for voivodeship in real_death_toll.index:
            row = real_death_toll.loc[voivodeship]
            first_death[voivodeship] = row[row >= n].index[0]
        
        return first_death
    
    # Get first day of pandemic based on in how many percent of counties there was at least one death case ***********
    @classmethod
    def get_starting_days_for_voivodeships_based_on_district_deaths(cls,
                                                                    percent_of_touched_counties,
                                                                    ignore_healthy_counties):
        """
        Returns dict sorted by values in which:
            key = voivodeship
            value = day since which pandemic fully started in given voivodeship.

        percent_of_touched_counties determines in how many counties has to die at least one pearson
        to say that pandemic in voivodeship started.format

        ignore_healthy_counties, some of the counties never had dead pearson, so if this variable is set to True
        then counties without dead are ignored while finding first day of pandemic for entire voivodeship.
        """
        df_excel = cls.__get_df_excel_deaths_early()
        counties_in_voivodeship = cls.get_counties_in_voivodeship()
        
        starting_day_for_voivodeship = {}
        for voivodeship, counties in counties_in_voivodeship.items():
            starting_days = []
            for district in counties:
                day = np.argmax(df_excel.loc[district] > 0)
                starting_days.append(day)
            
            if ignore_healthy_counties:
                starting_days = [day for day in starting_days if day > 0]
            else:
                starting_days = [day if day > 0 else max(starting_days) for day in starting_days]
            starting_days.sort()
            
            index = len(starting_days) * percent_of_touched_counties / 100
            index = int(round(index, 0))
            index = min(index, len(starting_days) - 1)
            
            starting_day_for_voivodeship[voivodeship] = starting_days[index]
        
        # sort dict by values of values (days increasing)
        starting_day_for_voivodeship = {k: v for k, v in
                                        sorted(starting_day_for_voivodeship.items(), key=lambda item: item[1])}
        return starting_day_for_voivodeship
    
    # Get death toll for each voivodeship from which pandemic started for good, threshold was set individually for
    # each voivodeship by hand i.e. by looking since how many deaths death tool looks nicely. ************************
    @classmethod
    def get_starting_deaths_by_hand(cls):
        """
        Returns dict in which key = voivodeship, value = since how many deaths death tool looks smooth.
        """
        result = {'dolnośląskie': 1,
                  'kujawsko-pomorskie': 100,
                  'lubelskie': 50,
                  'lubuskie': 50,
                  'łódzkie': 1,
                  'małopolskie': 200,
                  'mazowieckie': 1,
                  'opolskie': 100,
                  'podkarpackie': 200,
                  'podlaskie': 50,
                  'pomorskie': 50,
                  'śląskie': 1,
                  'świętokrzyskie': 100,
                  'warmińsko-mazurskie': 20,
                  'wielkopolskie': 1,
                  'zachodniopomorskie': 50}
        
        return result
    
    # Get death toll in day specified for each voivodeship by input dict like  {mazowieckie: 50, śląskie: 34, ...}
    @classmethod
    def get_starting_death_toll_for_voivodeships_by_days(cls, voivodeships_days):
        """
        Returns dict sorted by values in which:
            key = voivodeship
            value = death toll in that voivodeship in a day given by input dictionary.

        :param voivodeships_days: dict that specifies for what day death toll will be returned, example:
            voivodeships_days = {mazowieckie: 50, śląskie: 34, opolskie: 94, ...}
        """
        death_toll = cls.get_real_death_toll()
        
        starting_death_toll = {}
        for voivodeship, day in voivodeships_days.items():
            starting_death_toll[voivodeship] = death_toll.loc[voivodeship][day]
        
        # sort dict by values of values (death toll increasing)
        starting_death_toll = {k: v for k, v in sorted(starting_death_toll.items(), key=lambda item: item[1])}
        return starting_death_toll
    
    # Get death toll shifted (with respect to real days), such in given day there was not less than specified deaths *
    @classmethod
    def get_shifted_real_death_toll_to_common_start_by_num_of_deaths(cls, starting_day=3, minimum_deaths=1):
        """
        Returns df with indices = voivodeship_names and columns = [0, 1, 2, 3, ...] which represents
        next days of pandemic.

        For every voivodeship first non zero value occurs in column = starting_day.
        In that day death_toll was not less than ,,minimum_deaths''.
        """
        real_death_toll = cls.get_real_death_toll()
        
        # create new df (shifted_real_death_toll) in which
        # for every voivodeship first death occurred in day = starting_day ---------------------------
        days = len(real_death_toll.columns.to_list())
        shifted_real_death_toll = pd.DataFrame(columns=list(range(days)))
        for voivodeship in real_death_toll.index:
            x = real_death_toll.loc[voivodeship]
            x = x[x >= minimum_deaths].to_list()
            shifted_real_death_toll.loc[voivodeship] = pd.Series(x)
        
        shifted_real_death_toll.columns = np.array(shifted_real_death_toll.columns.to_list()) + starting_day
        shifted_real_death_toll.index.name = 'voivodeship'
        # ----------------------------------------------------------------------------------------------
        
        # for plotting reasons I want to have deaths = [(0,0), (1, 0), (2, 5), (3, 7), ...] instead of
        # [(0, 5), (1, 7), ...], so it is done below by expanding shifted_real_death_toll dataframe -----------------
        if starting_day > 0:
            # Create df of data for first n = starting_day days of pandemic
            # Columns = [1, 2, ..., n], indices = voivodeships, df filled with zeroes
            left_df = pd.DataFrame(columns=list(range(starting_day)))
            for voivodeship in real_death_toll.index:
                x = [0] * starting_day
                left_df.loc[voivodeship] = pd.Series(x)
            left_df.index.name = 'voivodeship'
            
            # extend shifted_real_death_toll to obtain zeroes for first  n = starting_day days of pandemic
            shifted_real_death_toll = left_df.merge(shifted_real_death_toll, on='voivodeship', how='inner')
        # -----------------------------------------------------------------------------------------------------------
        
        return shifted_real_death_toll
