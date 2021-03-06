"""Extracting real pandemic and geospatial data"""
import codecs
import pickle
import datetime
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.config import Directories, RealDataOptions, \
    ModelOptions, StartingDayBy
from disease_spread_model.model.my_math_utils import *


class RealData:
    """
    Purpose of this class is to provide following real data:
        * voivodeships() - list of voivodeships
        * get_real_general_data() - df with geographic data (unrelated to pandemic) for voivodeships
        * death_tolls() - df with death toll dynamic for voivodeships
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
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"geospatial/"
        f"GUS_general_pop_data.csv")

    __fname_counties_in_voivodeship_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"geospatial/"
        f"counties_in_voivodeship.pck")

    # http://eregion.wzp.pl/liczba-mieszkancow-przypadajacych-na-1-sklep
    __fname_peoples_per_shop = (
        f"{Directories.ABM_DIR}/"
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
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"pandemic/"
        f"early_data/"
        f"COVID-19_04.03-23.11.xlsm")

    __dir_pandemic_day_by_day_late_raw = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"raw/"
        f"pandemic/"
        f"late_data/")

    __fname_pandemic_day_by_day_late_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"pandemic_day_by_day.pck")

    __fname_entire_death_toll_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"entire_death_toll.pck")

    __fname_entire_interpolated_death_toll_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"entire_interpolated_death_toll.pck")

    __fname_entire_infected_toll_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"entire_infected_toll.pck")

    __fname_df_excel_deaths_pandemic_early_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"df_excel_deaths_pandemic_early_final.pck")

    __fname_df_excel_infections_pandemic_early_final = (
        f"{Directories.ABM_DIR}/"
        f"disease_spread_model/"
        f"data/"
        f"processed/"
        f"pandemic/"
        f"df_excel_infections_pandemic_early_final.pck")

    # Get voivodeships ************************************************************************************************
    @classmethod
    def voivodeships(cls):
        """ Returns list containing lowercase voivodeship names. """
        return cls.__voivodeships

    @staticmethod
    def day_to_date(day_number: Union[int, str]) -> str:
        """
        Return `2020-03-04 + day number` as str in format YYYY-MM-DD.

        If date (str) is passed than it is returned without any changes.

        Note: `datetime.timedelta` is messed up such that `datetime.timedelta(int)`
            works fine, but `datetime.timedelta(np.int64)` raises TypeError.
            So there is need to convert int to int.
        """
        if isinstance(day_number, str):
            return day_number
        else:
            day_number = int(day_number)

        date0 = datetime.datetime(2020, 3, 4)
        date0 += datetime.timedelta(days=day_number)
        return date0.strftime('%Y-%m-%d')

    @staticmethod
    def date_to_day(date: Union[int, str]) -> int:
        """
        Return `date - 2020-03-04` as int.

        If day (int) is passed than it is returned without any changes.

        Note: `datetime.timedelta` is messed up such that `datetime.timedelta(int)`
            works fine, but `datetime.timedelta(np.int64)` raises TypeError.
            So there is need to convert int to int.
        """
        if not isinstance(date, str):
            return date

        date0 = datetime.datetime(2020, 3, 4)
        return (datetime.datetime.strptime(date, "%Y-%m-%d") - date0).days

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

        voivodeships = cls.voivodeships()
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
        counties_in_voivodeship = {k.lower(): v for k, v in
                                   counties_in_voivodeship.items()}
        return counties_in_voivodeship

    @classmethod
    def __save_counties_in_voivodeship_as_pickle(cls):
        counties_in_voivodeship = cls.__get_counties_in_voivodeship()

        save_dir = os.path.split(cls.__fname_counties_in_voivodeship_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_counties_in_voivodeship_final, 'wb') as handle:
            pickle.dump(counties_in_voivodeship, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

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

    # Get data about population, population density, urbanization and
    # shops among voivodeships [2019 and 2021] *******
    @classmethod
    def get_real_general_data(
            cls,
            customers_in_household=ModelOptions.CUSTOMERS_IN_HOUSEHOLD):

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
        df = df.drop([0, 1])  # drop redundant columns

        # manually set column names, because they are broken
        df.columns = ['voivodeship',
                      'pop cities 2010', 'pop cities 2019',
                      'pop village 2010', 'pop village 2019',
                      'urbanization 2010', 'urbanization 2019']

        # drop not interesting columns
        df.drop(columns=['pop cities 2010',
                         'pop village 2010',
                         'urbanization 2010'],
                inplace=True)

        # make column values compatible with the accepted convention
        df['voivodeship'] = df['voivodeship'].str.lower()

        # Convert strings to numbers
        df['pop cities 2019'] = [int(item.replace(' ', '')) for item in
                                 df['pop cities 2019']]
        df['pop village 2019'] = [int(item.replace(' ', '')) for item in
                                  df['pop village 2019']]
        df['urbanization 2019'] = [float(item.replace(',', '.')) for item in
                                   df['urbanization 2019']]

        # Make new column having population of voivodeships
        df['population'] = df['pop cities 2019'] + df['pop village 2019']

        # drop not interesting columns
        df.drop(columns=['pop cities 2019', 'pop village 2019'], inplace=True)

        # set new names to columns as now there are
        # only data from 2019 (not 2010)
        df.columns = ['voivodeship', 'urbanization', 'population']

        # set voivodeship column as an index column
        df.set_index('voivodeship', inplace=True)

        # keep only those columns
        df = df[['population', 'urbanization']]
        # ------------------------------------------------------------

        # Get data about population density from GUS webpage [2021] --------
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
        df['population density'] = list(population_density_mixed.values())
        # -----------------------------------------------------------------

        # Put data about num of peoples per shop [2021] to new tmp DataFrame
        tmp_df = pd.read_csv(cls.__fname_peoples_per_shop)
        tmp_df.drop(0, inplace=True)
        tmp_df.rename(columns={'Województwa/Lata': 'voivodeship'},
                      inplace=True)
        tmp_df['voivodeship'] = tmp_df['voivodeship'].str.lower()
        tmp_df.set_index('voivodeship', inplace=True)

        # Get Series containing data about
        # number of peoples per shop for all voivodeships
        shop_series = pd.Series(tmp_df['2019'], name='peoples per shop')

        # Merge previous DataFrame with peoples per shop Series
        df = pd.concat([df, shop_series], axis=1)

        # Determine N and grid size
        # based on population and peoples per shop
        shops = df['population'] / df['peoples per shop']
        shops_model = shops / 20

        # add data about number of shops to DataFrame
        df['shops'] = shops.astype(int)
        df['shops MODEL'] = shops_model.astype(int)

        # add column with grid side length such that:
        # grid_side_length**2 = rescaled num of shops in voivodeship
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
        day = f'{fname_only[:4]}-{fname_only[4:6]}-{fname_only[6:8]}'

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
        fnames = all_fnames_from_dir(
            directory=cls.__dir_pandemic_day_by_day_late_raw)

        # create empty Dataframes having a desired format (same like read dataframe ,,df'')
        df = cls.__get_significant_pandemic_late_data(fname=fnames[0])
        voivodeships = df.index.to_list()
        cols = df.columns.to_list()
        result_dict = {
            voivodeship: pd.DataFrame(columns=cols) for voivodeship in
            voivodeships
        }

        # one day of pandemic is one file so iterate over them, grab data and insert as rows to Dataframes
        for fname in fnames:
            df = cls.__get_significant_pandemic_late_data(fname=fname)
            # modify result voivodeships adding read data, next file = new row
            for voivodeship in voivodeships:
                voivodeship_df = result_dict[voivodeship]

                voivodeship_df.loc[-1] = df.loc[voivodeship,
                                         :].values  # adding a row
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

        save_dir = os.path.split(cls.__fname_pandemic_day_by_day_late_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_pandemic_day_by_day_late_final, 'wb') as handle:
            pickle.dump(real_late_data, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

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

        valid_rows = [voivodeship.upper() for voivodeship in
                      cls.voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=[158, 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)

        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)

        return df_excel.max(axis=1)

    @classmethod
    def __get_death_toll_for_early_pandemic(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma zgonów'

        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)

        valid_rows = [voivodeship.upper() for voivodeship in
                      cls.voivodeships()]
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
        late_pandemic_death_toll = pd.DataFrame(
            columns=['voivodeship'] + late_days)
        late_pandemic_death_toll.set_index('voivodeship', inplace=True)

        death_toll_at_end_of_early_stage = cls.__get_death_toll_at_end_of_early_pandemic()
        for voivodeship, df in late_pandemic_data.items():
            if voivodeship != 'Cały kraj':
                late_pandemic_death_toll.loc[voivodeship] = \
                    np.cumsum(df['zgony'].to_list()) + \
                    death_toll_at_end_of_early_stage[voivodeship]

        late_pandemic_death_toll = late_pandemic_death_toll.astype(int)

        early_pandemic_death_toll = cls.__get_death_toll_for_early_pandemic()

        return early_pandemic_death_toll.merge(
            late_pandemic_death_toll, on='voivodeship', how='inner'
        )

    @classmethod
    def __save_entire_death_toll_as_pickle(cls):
        entire_death_toll = cls.__merge_properly_early_and_late_pandemic_death_toll()

        save_dir = os.path.split(cls.__fname_entire_death_toll_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_entire_death_toll_final, 'wb') as handle:
            pickle.dump(entire_death_toll, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def __load_entire_death_toll_from_pickle(cls):
        with open(cls.__fname_entire_death_toll_final, 'rb') as handle:
            entire_death_toll = pickle.load(handle)
        return entire_death_toll

    @classmethod
    def death_tolls(cls):
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

        valid_rows = [voivodeship.upper() for voivodeship in
                      cls.voivodeships()]
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

        valid_rows = [voivodeship.upper() for voivodeship in
                      cls.voivodeships()]
        df_excel = df_excel.loc[df_excel['Nazwa'].isin(valid_rows)]
        df_excel.drop(columns=['Kod', 'Unnamed: 1'], inplace=True)
        df_excel['Nazwa'] = [name.lower() for name in df_excel['Nazwa']]
        df_excel.rename(columns={'Nazwa': 'voivodeship'}, inplace=True)
        df_excel.set_index('voivodeship', inplace=True)

        dates = pd.date_range(start='2020-03-04', end='2020-11-24').tolist()
        dates = [f'{i.year:04d}-{i.month:02d}-{i.day:02d}' for i in dates]
        df_excel.columns = dates
        df_excel.drop(columns=['2020-11-24'], inplace=True)

        return df_excel.max(axis=1)

    @classmethod
    def __merge_properly_early_and_late_pandemic_infected_toll(cls):
        late_pandemic_data = cls.__get_real_late_pandemic_data()

        late_days = late_pandemic_data['Cały kraj']['day'].to_list()
        late_pandemic_infected_toll = pd.DataFrame(
            columns=['voivodeship'] + late_days)
        late_pandemic_infected_toll.set_index('voivodeship', inplace=True)

        infected_toll_at_end_of_early_stage = cls.__get_infected_toll_at_end_of_early_pandemic()
        for voivodeship, df in late_pandemic_data.items():
            if voivodeship != 'Cały kraj':
                late_pandemic_infected_toll.loc[voivodeship] = \
                    np.cumsum(df['liczba_przypadkow'].to_list()) + \
                    infected_toll_at_end_of_early_stage[voivodeship]

        late_pandemic_infected_toll = late_pandemic_infected_toll.astype(int)

        early_pandemic_infected_toll = cls.__get_infected_toll_for_early_pandemic()

        return early_pandemic_infected_toll.merge(
            late_pandemic_infected_toll, on='voivodeship', how='inner'
        )

    @classmethod
    def __save_entire_infected_toll_as_pickle(cls):
        entire_infected_toll = cls.__merge_properly_early_and_late_pandemic_infected_toll()

        save_dir = os.path.split(cls.__fname_entire_infected_toll_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_entire_infected_toll_final, 'wb') as handle:
            pickle.dump(entire_infected_toll, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

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

    # Get data about recovered toll
    # infected toll (shifted) - death toll --------------------------
    @classmethod
    def recovered_tolls(cls):
        """
        Get recovered toll as infected toll shifted by 14 days minus
        death toll.
        """

        def shift_data_by_n_days(df, n):
            org_dates = [datetime.datetime.strptime(str(_date), "%Y-%m-%d") for
                         _date in df.columns]
            shift_dates = [_date + datetime.timedelta(days=n) for _date in
                           org_dates]
            df.columns = shift_dates
            return df

        # get shifted infected toll (missing data for first days)
        infected_toll = cls.get_real_infected_toll()
        shifted_infected_toll = shift_data_by_n_days(df=infected_toll, n=14)

        # Fill missing data for first days in shifted infected toll
        # Create dict which will be converted to df
        d = {}
        num_of_voivodeships = len(RealData.voivodeships())
        for day_num in range(14):
            date = datetime.datetime(2020, 3, 4) + datetime.timedelta(
                days=day_num)
            d[date] = np.zeros(num_of_voivodeships)

        # fill missing data for first days in shifted infected toll
        artificial_infected_toll_first_days = pd.DataFrame(data=d)
        artificial_infected_toll_first_days.set_index(
            pd.Index(cls.voivodeships()),
            inplace=True)

        # append missing first columns to shifted infected toll
        shifted_infected_toll = pd.concat([artificial_infected_toll_first_days,
                                           shifted_infected_toll],
                                          axis=1, join="inner")

        # get death toll and convert column string labels to datetime objects
        death_toll = cls.death_tolls()
        death_toll = shift_data_by_n_days(df=death_toll, n=0)

        # make df of recovered toll
        recovered_toll = shifted_infected_toll.sub(death_toll, fill_value=0)

        return recovered_toll

    # Get data about death toll in it's early stage (before data from GUS)
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

        save_dir = \
            os.path.split(cls.__fname_df_excel_deaths_pandemic_early_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_df_excel_deaths_pandemic_early_final,
                  'wb') as handle:
            pickle.dump(df_excel_deaths_pandemic_early, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def __load_df_excel_deaths_pandemic_early_from_pickle(cls):
        with open(cls.__fname_df_excel_deaths_pandemic_early_final,
                  'rb') as handle:
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

    # Get data about infected toll in it's early stage (before data from GUS)
    @classmethod
    def __get_df_excel_infections_pandemic_early(cls):
        io = cls.__fname_pandemic_day_by_day_early
        sheet_name = 'Suma przypadków'

        df_excel = pd.read_excel(io=io, sheet_name=sheet_name)

        df_excel.drop(columns=['Kod', "Unnamed: 1"], inplace=True)
        df_excel.drop([0, 1], inplace=True)
        df_excel.set_index('Nazwa', inplace=True)

        # rop last col (24.11.2019) as it is empty
        df_excel = df_excel.iloc[:, :-1]

        # some counties have the same name which causes problem so I'll rename them ...
        df_excel = rename_duplicates_in_df_index_column(df_excel)

        return df_excel

    @classmethod
    def __save_df_excel_infections_pandemic_early_as_pickle(cls):
        df_excel_infections_pandemic_early = cls.__get_df_excel_infections_pandemic_early()

        save_dir = \
            os.path.split(
                cls.__fname_df_excel_infections_pandemic_early_final)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(cls.__fname_df_excel_infections_pandemic_early_final,
                  'wb') as handle:
            pickle.dump(df_excel_infections_pandemic_early, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def __load_df_excel_infections_pandemic_early_from_pickle(cls):
        with open(cls.__fname_df_excel_infections_pandemic_early_final,
                  'rb') as handle:
            df_excel_infections_pandemic_early = pickle.load(handle)
        return df_excel_infections_pandemic_early

    @classmethod
    def _get_df_excel_infections_early(cls):
        """
        Returns data from excel file about infections in early stage of pandemic as
        pandas DataFrame
        """
        try:
            df_excel_infections_pandemic_early = cls.__load_df_excel_infections_pandemic_early_from_pickle()
        except FileNotFoundError:
            cls.__save_df_excel_infections_pandemic_early_as_pickle()
            df_excel_infections_pandemic_early = cls.__load_df_excel_infections_pandemic_early_from_pickle()

        return df_excel_infections_pandemic_early

    # Get first day of pandemic based on number of deaths ************************************************************
    @classmethod
    def get_date_of_first_n_death(cls, n) -> dict[str: str]:
        """
        This function returns dict in which keys are voivodeship names (lowercase) and
        value is a day (YYYY-MM-DD) of first death in given voivodeship.

        Useful function to define beginning of pandemic in given voivodeship.
        """
        real_death_toll = cls.death_tolls()
        first_death = {}
        for voivodeship in real_death_toll.index:
            row = real_death_toll.loc[voivodeship]
            first_death[voivodeship] = row[row >= n].index[0]

        return first_death

    # Get first day of pandemic based on in how many percent of counties there was at least one death case ***********
    @classmethod
    def _get_starting_days_for_voivodeships_based_on_district_deaths(
            cls,
            percent_of_touched_counties=RealDataOptions.PERCENT_OF_DEATH_COUNTIES,
            ignore_healthy_counties=False):

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
                starting_days = [day if day > 0 else max(starting_days) for day
                                 in starting_days]
            starting_days.sort()

            index = len(starting_days) * percent_of_touched_counties / 100
            index = int(round(index, 0))
            index = min(index, len(starting_days) - 1)

            starting_day_for_voivodeship[voivodeship] = starting_days[index]

        # Return sorted dict by values (days increasing)
        return sort_dict_by_values(starting_day_for_voivodeship)

    # Get first day of pandemic based on in how many percent of counties there was at least one infected case ********
    @classmethod
    def _get_starting_days_for_voivodeships_based_on_district_infections(
            cls,
            percent_of_touched_counties=RealDataOptions.PERCENT_OF_INFECTED_COUNTIES):
        """
        Returns dict sorted by values in which:
            key = voivodeship
            value = day since which pandemic fully started in given voivodeship.

        percent_of_touched_counties determines in how many counties has to become infected
        at least one pearson to say that pandemic in voivodeship started.

        ignore_healthy_counties, some of the counties never had infected pearson, so if this variable is set to True
        then counties without infections are ignored while finding first day of pandemic for entire voivodeship.
        """
        df_excel = cls._get_df_excel_infections_early()
        counties_in_voivodeship = cls.get_counties_in_voivodeship()

        starting_day_for_voivodeship = {}
        for voivodeship, counties in counties_in_voivodeship.items():

            # find starting days for each counties as a day when someone got infected
            starting_days = []
            for district in counties:
                day = np.argmax(df_excel.loc[district] > 0)
                starting_days.append(day)

            # if no one got infected --> day = last day
            starting_days = [day if day > 0 else max(starting_days) for day in
                             starting_days]
            starting_days.sort()

            # find day number in which in given percent of counties someone got infected
            index = len(starting_days) * percent_of_touched_counties / 100
            index = int(round(index, 0))
            index = min(index, len(starting_days) - 1)

            # add result to returning dict
            starting_day_for_voivodeship[voivodeship] = starting_days[index]

        # Return sorted dict by values (days increasing)
        return sort_dict_by_values(starting_day_for_voivodeship)

    @classmethod
    def starting_days(
            cls,
            by=RealDataOptions.STARTING_DAYS_BY,
            percent_of_touched_counties=RealDataOptions.PERCENT_OF_INFECTED_COUNTIES,
            ignore_healthy_counties=False):

        # get starting day of pandemic by percent of touched counties
        if by == StartingDayBy.DEATHS:
            starting_days = \
                cls._get_starting_days_for_voivodeships_based_on_district_deaths(
                    percent_of_touched_counties=percent_of_touched_counties,
                    ignore_healthy_counties=ignore_healthy_counties)
        elif by == StartingDayBy.INFECTIONS:
            starting_days = \
                cls._get_starting_days_for_voivodeships_based_on_district_infections(
                    percent_of_touched_counties=percent_of_touched_counties)
        else:
            raise NotImplementedError

        return starting_days

    # Get death toll for each voivodeship from which pandemic started for good, threshold was set individually for
    # each voivodeship by hand i.e. by looking since how many deaths death tool looks nicely. ************************
    @classmethod
    def get_starting_deaths_by_hand(cls):
        """
        Returns dict in which key = voivodeship, value = since how many deaths death tool looks smooth.
        """
        return {'dolnośląskie': 1,
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

    # Get death toll in day specified for each voivodeship by input dict like  {mazowieckie: 50, śląskie: 34, ...}
    @classmethod
    def get_starting_death_toll_for_voivodeships_by_days(cls,
                                                         voivodeships_days):
        """
        Returns dict sorted by values in which:
            key = voivodeship
            value = death toll in that voivodeship in a day given by input dictionary.

        :param voivodeships_days: dict that specifies for what day death toll will be returned, example:
            voivodeships_days = {mazowieckie: 50, śląskie: 34, opolskie: 94, ...}
        """
        death_toll = cls.death_tolls()

        starting_death_toll = {
            voivodeship: death_toll.loc[voivodeship][day]
            for voivodeship, day in voivodeships_days.items()
        }

        # Return sorted dict by values of values (death toll increasing)
        return sort_dict_by_values(starting_death_toll)

    # Get death toll shifted (with respect to real days), such in given day there was not less than specified deaths *
    @classmethod
    def get_shifted_real_death_toll_to_common_start_by_num_of_deaths(cls,
                                                                     starting_day=3,
                                                                     minimum_deaths=1):
        """
        Returns df with indices = voivodeship_names and columns = [0, 1, 2, 3, ...] which represents
        next days of pandemic.

        For every voivodeship first non zero value occurs in column = starting_day.
        In that day death_toll was not less than ,,minimum_deaths''.
        """
        real_death_toll = cls.death_tolls()

        # create new df (shifted_real_death_toll) in which
        # for every voivodeship first death occurred in day = starting_day ---------------------------
        days = len(real_death_toll.columns.to_list())
        shifted_real_death_toll = pd.DataFrame(columns=list(range(days)))
        for voivodeship in real_death_toll.index:
            x = real_death_toll.loc[voivodeship]
            x = x[x >= minimum_deaths].to_list()
            shifted_real_death_toll.loc[voivodeship] = pd.Series(x)

        shifted_real_death_toll.columns = np.array(
            shifted_real_death_toll.columns.to_list()) + starting_day
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
            shifted_real_death_toll = left_df.merge(shifted_real_death_toll,
                                                    on='voivodeship',
                                                    how='inner')
        # -----------------------------------------------------------------------------------------------------------

        return shifted_real_death_toll

    # Get last day of pandemic for each voivodeship
    @staticmethod
    def _get_neighbouring_indices_with_max_delta_value(data: np.ndarray):
        """
        Returns tuple (idx, idx + 1) for which expression
        'abs(data[idx + 1] - data[idx])' takes the maximum.

        Example: [1, 2, 8, 1, 5, 6, 3] --> (2, 3)
            as arr[3] - arr[2] = 1 - 8 = -7 <--- max delta

        :param data: array of numeric values
        :type data: np.ndarray
        """

        max_delta = 0
        idx = None
        for i in range(len(data) - 1):
            val1 = data[i]
            val2 = data[i + 1]

            delta = abs(val2 - val1)
            if delta > max_delta:
                max_delta = delta
                idx = i

        return idx, idx + 1

    @staticmethod
    def bad_voivodeships() -> list[str]:
        return ['lubuskie', 'podkarpackie', 'warmińsko-mazurskie']

    @classmethod
    def ending_days_by_death_toll_slope(
            cls,
            starting_days_by=RealDataOptions.STARTING_DAYS_BY,
            percent_of_touched_counties=RealDataOptions.PERCENT_OF_INFECTED_COUNTIES,
            last_date=RealDataOptions.LAST_DATE,
            death_toll_smooth_out_win_size=RealDataOptions.DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_WIN_SIZE,
            death_toll_smooth_out_polyorder=RealDataOptions.DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_SAVGOL_POLYORDER,
    ):  # sourcery no-metrics
        """
        Returns last day of first phase of pandemic.
        
        Last day is the closest day to two month since start of pandemic,
        with a death toll slope peak at least as high as a half of max
        death toll slope.
        
        If this function is called again with same args as ever before
        then return data from file created before
        
        Algorithm:
            - get death toll and fill missing data in it
            - smooth out death toll
            - fit line for each segment of smooth 'death_toll[day-3: day+4]'
            - normalize slope such 'max_slope=1' on given interval
            - plot slope (aka. derivative)
            - find peaks in a slope (up and down)
            - find last day as peak:
                * nearest to day 60 since start day
                * peak value > 0.5
        
        :param starting_days_by: 'deaths' or 'infections'
        :type starting_days_by: StartingDayBy
        :param percent_of_touched_counties: how many counties has to be
            affected by 'start_days_by' to set given day as first day of pandemic.
        :type percent_of_touched_counties: int
        :param last_date: date in a format 'YYYY-mm-dd' that can possibly be the
            last day of pandemic
        :type last_date: str
        :param death_toll_smooth_out_win_size: window size used to smooth up
            death toll in 'savgol_filter', has to be odd number
        :type death_toll_smooth_out_win_size: int
        :param death_toll_smooth_out_polyorder: polyorder used for smooth up
            death toll in savgol_filter.
        :type death_toll_smooth_out_polyorder: int
        :return: dict {voivodeship: last_day_of_pandemic} for all voivodeships.
        :rtype: dict

        """

        def fdir_from_args():

            _dir = (f"{Directories.ABM_DIR}/"
                    f"disease_spread_model/"
                    f"data/"
                    f"processed/"
                    f"pandemic/"
                    f"last_days/")

            fname = (f"{starting_days_by}_{percent_of_touched_counties} "
                     f"last_date_{last_date} "
                     f"win_size_{death_toll_smooth_out_win_size}"
                     f"polyorder_{death_toll_smooth_out_polyorder}"
                     f".pck")

            return _dir + fname

        def save_result(result_dict):
            fdir = fdir_from_args()
            save_dir = os.path.split(fdir)[0]
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            with open(fdir, 'wb') as handle:
                pickle.dump(result_dict,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        def was_found_before():
            return os.path.isfile(fdir_from_args())

        def read_result():
            with open(fdir_from_args(), 'rb') as handle:
                return pickle.load(handle)

        def compute_last_days():

            # create empty result dict
            result = {}

            # get start days dict {voivodeship: days}
            start_days = cls.starting_days(
                by=starting_days_by,
                percent_of_touched_counties=percent_of_touched_counties,
                ignore_healthy_counties=False
            )

            # get real death toll for all voivodeships (since 03.04.2019)
            death_tolls = RealData.death_tolls()

            # fill gaps in real death toll
            for voivodeship in RealData.voivodeships():
                death_tolls.loc[voivodeship] = complete_missing_data(
                    values=death_tolls.loc[voivodeship])

            # smooth out death toll
            death_tolls_smooth = death_tolls.copy()
            for voivodeship in RealData.voivodeships():
                death_tolls_smooth.loc[voivodeship] = savgol_filter(
                    x=death_tolls.loc[voivodeship],
                    window_length=death_toll_smooth_out_win_size,
                    polyorder=death_toll_smooth_out_polyorder)

            # get last day in which death pandemic last day will be looked for
            last_day_to_search = list(death_tolls.columns).index(last_date)

            # find last day of pandemic for each voivodeship
            for voivodeship in cls.voivodeships():

                # get start day based on percent counties dead
                day0 = start_days[voivodeship]
                if day0 > last_day_to_search:
                    result[voivodeship] = np.NaN
                    continue

                # slope of smooth death toll
                slope_smooth = slope_from_linear_fit(
                    data=death_tolls_smooth.loc[voivodeship],
                    half_win_size=3)

                # normalize slope to 1
                if max(slope_smooth[day0: last_day_to_search]) > 0:
                    slope_smooth /= max(slope_smooth[day0: last_day_to_search])

                """Last day of pandemic as day of peak closest to 2 months 
                with derivative > 0.5."""
                vec = np.copy(slope_smooth[day0: last_day_to_search])
                last_day_candidates = find_peaks(
                    vec, distance=8, height=.5)[0]

                # No candidates --> (end day = start day)
                if len(last_day_candidates) == 0:
                    last_day_candidates = np.array([day0])

                # choose find last day (nearest to 60) from candidates
                last_day_since_day0 = min(last_day_candidates,
                                          key=lambda x: abs(x - 60))

                # Add 7 days, because death toll used to change rapidly
                # 7 days right after max death toll slope
                last_day_since_day0 += 7

                # add last day to result dict
                result[voivodeship] = day0 + last_day_since_day0

            return result

        if was_found_before():
            return read_result()
        else:
            last_days = compute_last_days()
            save_result(result_dict=last_days)
            return last_days

    @classmethod
    def fraction_of_infected_people(cls,
                                    voivodeship: str,
                                    day: Union[int, str]) -> float:
        """
        Return '(num of infected) / (num of all people)' in given voivodeship and day.
        Day 0 = 04.03.2020.
        """
        infected_tolls = cls.get_real_infected_toll()
        infected_toll = infected_tolls.loc[voivodeship]
        date = RealData.day_to_date(day)

        geospatial_df = cls.get_real_general_data()
        population = geospatial_df.loc[voivodeship]['population']

        return infected_toll[date] / population

    @classmethod
    def fraction_of_dead_people(cls,
                                voivodeship: str,
                                day: Union[int, str]) -> float:
        """
        Return '(num of infected) / (num of all people)' in given voivodeship and day.
        Day 0 = 04.03.2020.
        """
        death_tolls = cls.death_tolls()
        death_toll = death_tolls.loc[voivodeship]
        date = RealData.day_to_date(day)

        geospatial_df = cls.get_real_general_data()
        population = geospatial_df.loc[voivodeship]['population']

        return death_toll[date] / population

    @classmethod
    def estimated_fraction_of_init_infected_by_deaths(
            cls,
            starting_days_by: StartingDayBy,
            percent_of_touched_counties: int,
            mortality: float,

    ) -> pd.Series:

        geospatial_df = RealData.get_real_general_data()
        death_tolls = RealData.death_tolls()
        starting_days = RealData.starting_days(
            by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
        )

        estimated_values = {}
        for voivodeship in cls.__voivodeships:
            date = RealData.day_to_date(starting_days[voivodeship] + 7)
            death_toll = death_tolls.loc[voivodeship]
            num_of_deaths = death_toll[date]

            estimated_num_of_infected = num_of_deaths / mortality
            population = geospatial_df.loc[voivodeship]['population']
            estimated_frac_of_infected = (estimated_num_of_infected
                                          / population) * .25

            estimated_values[voivodeship] = estimated_frac_of_infected

        return pd.Series(
            estimated_values,
            name='initial fraction of infected customers')

    @classmethod
    def prepare_summary_table(
            cls,
            starting_days_by: RealDataOptions.STARTING_DAYS_BY,
            percent_of_touched_counties: int,
            last_date: str,
            death_toll_smooth_out_win_size: int,
            death_toll_smooth_out_polyorder: int,
    ) -> None:

        def format_values(ser: pd.Series, decimals: int,
                          name: str) -> pd.Series:
            return pd.Series(
                {voivodeship: '{0:.{1}f}'.format(ser[voivodeship], decimals)
                 for voivodeship in RealData.voivodeships()},
                name=name,
            )

        general_data = RealData.get_real_general_data(
            customers_in_household=3)

        lbrk = 'linebreak'

        populacja = format_values(
            ser=general_data['population'] / 1e6,
            decimals=2,
            name='populacja' + lbrk + '(mln)'
        )

        wsp_urbanizacji = format_values(
            ser=general_data['urbanization'],
            decimals=1,
            name='współczynnik' + lbrk + 'urbanizacji',
        )

        ludzi_na_sklep = format_values(
            ser=general_data['peoples per shop'],
            decimals=0,
            name='ilość osób' + lbrk + 'na 1 sklep'
        )

        sklepow = format_values(
            ser=general_data['shops'],
            decimals=0,
            name='ilość' + lbrk + 'sklepów'
        )

        sklepow_M = format_values(
            ser=general_data['shops MODEL'],
            decimals=0,
            name='docelowa ilość' + lbrk + 'powiatów'
        )

        gs = general_data['grid side MODEL'].to_dict()
        rozmiar_siatki_M = pd.Series(
            {k: f"({v} x {v})" for k, v in gs.items()},
            name='rozmiar' + lbrk + 'siatki',
        )

        domow_w_powiecie_M = pd.Series(
            general_data['N MODEL'],
            name='N - ilość domów' + lbrk + 'w powiecie')

        abs_blad_populacji = (
            general_data['population']
            - (general_data['grid side MODEL'] ** 2) * general_data['N MODEL'] * 3)

        procentowy_blad_populacji = format_values(
            ser=(abs_blad_populacji / general_data['population']) * 100,
            decimals=3,
            name='względny błąd' + lbrk + 'populacji (%)',
        )

        dni_poczatkowe = RealData.starting_days(
            by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
        )
        daty_poczatkowe = pd.Series(
            {voivodeship: RealData.day_to_date(day) for
             voivodeship, day in dni_poczatkowe.items()},
            name=('pierwszy dzień' + lbrk +
                  'początkowej fazy' + lbrk +
                  'pandemii'),
        )

        dni_koncowe = RealData.ending_days_by_death_toll_slope(
            starting_days_by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date=last_date,
            death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
            death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
        )
        daty_koncowe = pd.Series(
            {voivodeship: RealData.day_to_date(day) for
             voivodeship, day in dni_koncowe.items()},
            name=('ostatni dzień' + lbrk +
                  'początkowej fazy' + lbrk +
                  'pandemii'),
        )

        dlugosc_poczatkowego_etapu = pd.Series(
            {voivodeship:
                 dni_koncowe[voivodeship] - dni_poczatkowe[voivodeship]
             for voivodeship in dni_poczatkowe},
            name=('czas trwania' + lbrk +
                  'początkowej fazy' + lbrk +
                  'pandemii (w dniach)')
        )

        poczatkowy_etap_pandemii_dni = pd.Series(
            {voivodeship:
                 f'{daty_poczatkowe.to_dict()[voivodeship]} : '
                 f'{daty_koncowe.to_dict()[voivodeship]}, '
                 f'{dlugosc_poczatkowego_etapu.to_dict()[voivodeship]} dni'
             for voivodeship in RealData.voivodeships()
             },
            name='poczatkowy etap pandemii',
        )

        death_tolls = RealData.death_tolls()

        poczatkowa_ilosc_zgonow = pd.Series(
            {voivodeship:
                 death_tolls.loc[voivodeship][daty_poczatkowe[voivodeship]]
             for voivodeship in dni_poczatkowe})

        poczatkowa_ilosc_zgonow = format_values(
            ser=poczatkowa_ilosc_zgonow,
            decimals=0,
            name=('całkowita raportowa' + lbrk +
                  'ilość zgonów ' + lbrk +
                  'w pierwszym dniu' + lbrk +
                  'początkowej fazy pandemii'),
        )

        koncowa_ilosc_zgonow = pd.Series(
            {voivodeship:
                 death_tolls.loc[voivodeship][daty_koncowe[voivodeship]]
             for voivodeship in dni_poczatkowe})

        koncowa_ilosc_zgonow = format_values(
            ser=koncowa_ilosc_zgonow,
            decimals=0,
            name=('całkowita raportowa' + lbrk +
                  'ilość zgonów ' + lbrk +
                  'w ostatnim dniu' + lbrk +
                  'początkowej fazy pandemii'),
        )

        poczatkowy_etap_pandemii_zgony = pd.Series(
            {voivodeship:
                 f'od {poczatkowa_ilosc_zgonow.to_dict()[voivodeship]} '
                 f'do {koncowa_ilosc_zgonow.to_dict()[voivodeship]}'
             for voivodeship in RealData.voivodeships()
             },
            name=('suma zgonów' + lbrk +
                  'w początkowym' + lbrk +
                  'etapie pandemii'),
        )

        poczatkowy_procent_chorych_klientow = pd.Series(
            RealData.estimated_fraction_of_init_infected_by_deaths(
                starting_days_by=starting_days_by,
                percent_of_touched_counties=percent_of_touched_counties,
                mortality=2 / 100,
            ) * 100)

        poczatkowy_procent_chorych_klientow = format_values(
            ser=poczatkowy_procent_chorych_klientow,
            decimals=5,
            name=('początkowy odsetek' + lbrk +
                  'zarażonych klientów' + lbrk +
                  'do całej populacji (%)'),
        )

        real_summary_df = pd.DataFrame([
            populacja, wsp_urbanizacji, ludzi_na_sklep, sklepow
        ]).transpose()

        real_calc_summary_df = pd.DataFrame([
            poczatkowy_etap_pandemii_dni,
            poczatkowy_etap_pandemii_zgony,
            poczatkowy_procent_chorych_klientow,
        ]).transpose()

        model_values_summary_df = pd.DataFrame([
            sklepow_M, rozmiar_siatki_M, domow_w_powiecie_M,
            procentowy_blad_populacji
        ]).transpose()

        summary_df = pd.DataFrame([
            populacja, ludzi_na_sklep, wsp_urbanizacji, sklepow, sklepow_M,
            rozmiar_siatki_M, domow_w_powiecie_M,
            daty_poczatkowe, daty_koncowe, dlugosc_poczatkowego_etapu,
            poczatkowa_ilosc_zgonow, koncowa_ilosc_zgonow,
            poczatkowy_procent_chorych_klientow,
        ]).transpose()

        save_text(
            fdir=Directories.LATEX_TABLE_DIR + 'all_summary_df.tex',
            text=DfToLatex.df_to_latex(df=summary_df, lbrk=lbrk)
        )

        save_text(
            fdir=Directories.LATEX_TABLE_DIR + 'real_summary_df.tex',
            text=DfToLatex.df_to_latex(df=real_summary_df, lbrk=lbrk)
        )

        save_text(
            fdir=Directories.LATEX_TABLE_DIR + 'real_clac_summary_df.tex',
            text=DfToLatex.df_to_latex(df=real_calc_summary_df, lbrk=lbrk)
        )

        save_text(
            fdir=Directories.LATEX_TABLE_DIR + 'model_vals_summary_df.tex',
            text=DfToLatex.df_to_latex(df=model_values_summary_df, lbrk=lbrk)
        )


def main():
    """Can do some on fly tests/actions here."""

    # r = RealData.get_real_general_data()
    # r = RealData.get_real_infected_toll()

    RealData.prepare_summary_table(
        starting_days_by=StartingDayBy.INFECTIONS,
        percent_of_touched_counties=80,
        last_date='2020-07-01',
        death_toll_smooth_out_win_size=21,
        death_toll_smooth_out_polyorder=3
    )


if __name__ == '__main__':
    main()
