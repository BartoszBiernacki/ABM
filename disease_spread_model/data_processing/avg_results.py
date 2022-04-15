import shutil

from .real_data import RealData
from .text_processing import *
from disease_spread_model.config import Directories, ModelOptions
from disease_spread_model.names import TRANSLATE


class Results(object):
    def __init__(self):
        pass

    @classmethod
    def get_single_results(cls,
                           not_avg_data_directory=Directories.TMP_SAVE_DIR):
        """
        Returns dict in which:
            keys are tuples of variable_params and run
            values are result dataframes (one dataframe = one run of simulation)

        Intended to plot stochastic of simulations.
        """
        fnames = all_fnames_from_dir(directory=not_avg_data_directory)
        unique_tuples = get_list_of_tuples_from_dir(
            directory=not_avg_data_directory)

        return {
            unique_tuples[i]: pd.read_pickle(fname)
            for i, fname in enumerate(fnames)
        }

    @classmethod
    def get_avg_results(cls,
                        variable_params,
                        ignore_dead_pandemics=False):
        """
        Returns dict in which:
            keys are tuples of variable_params
            values are dataframes averaged over all simulation iterations
            
        Grabs all results from single simulations (in TMP_SAVE/).
        They are named in specific way: the name is a tuple that stores
        all the information about model parameters and last elem of tuple is a
        number of the simulation performed. Thanks to that simulations can be
        grouped so in one group there are simulations that had the same model parameters.
        Then groups of simulations are being averaged and returned.
        
        Function allows to ignore expired pandemics when calculating average.
        Pandemic is considered as expired when in the last day of simulation
        there are not any infected or prodromal cashiers and clients.
        """

        data_collector_model_results = Results.get_single_results()

        num_of_variable_model_params = len(variable_params)
        list_of_tuples = get_list_of_tuples_from_dir(
            directory=Directories.TMP_SAVE_DIR)
        tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples,
                                               start_length=num_of_variable_model_params)

        result = {}
        for key in tuples_grouped.keys():  # key is a tuple by which other tuples were grouped. For example key=(5, 2, 2)
            lis = []
            for item in tuples_grouped[
                key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
                df = data_collector_model_results[item]
                if ignore_dead_pandemics:
                    last_idx = len(df.index) - 1

                    incubation_people = df['Incubation people'][last_idx]
                    prodromal_people = df['Prodromal people'][last_idx]
                    illness_people = df['Illness people'][last_idx]
                    incubation_cashiers = df['Incubation cashiers'][last_idx]
                    prodromal_cashiers = df['Prodromal cashiers'][last_idx]

                    if incubation_people + prodromal_people + illness_people + \
                            incubation_cashiers + prodromal_cashiers != 0:
                        lis.append(
                            df)  # list of results dataframes matching key_tuple=(5, 2, 2)
                else:
                    lis.append(
                        df)  # list of results dataframes matching key_tuple=(5, 2, 2)

            if lis:
                array_with_all_iterations_results_for_specific_parameters = np.array(
                    lis)

                average_array = np.mean(
                    array_with_all_iterations_results_for_specific_parameters,
                    axis=0)
                df = pd.DataFrame(data=average_array)
                df.columns = data_collector_model_results[
                    list_of_tuples[0]].columns

                result[key] = df

        return result

    @classmethod
    def save_avg_results(cls,
                         avg_results,
                         fixed_params,
                         variable_params,
                         runs,
                         base_params=(),
                         include_voivodeship=True):
        """
        Save averaged simulation results to many files.
        
        :param avg_results: dict containing averaged data to save, details:
            key is a tuple containing values of variable params by which data were averaged.
            value is a DataFrame containing averaged data from many simulations, which were started with the
            variable parameters as in key.
        :type avg_results: dictionary
        :param fixed_params: fixed_params passed to run the model.
        :type fixed_params: dict
        :param variable_params: variable_params passed to run the model.
        :type variable_params: dictionary
        :param runs: number of simulation iterations for given set of parameters.
        :type runs: integer.
        :param base_params: names of model parameters, which values will be included in resulting folder name.
        :type base_params: tuple(str)
        :param include_voivodeship: include voivodeship in folder name if number of households in simulation matches
            number of households in some voivodeship?
        :type include_voivodeship: boolean
        
        :return: directory to saved data
        :rtype: string
        """

        def _make_all_params_shorten_dict() -> dict:
            """Merge all params in one dict
            (merge also default not given params)."""

            all_params = {**fixed_params, **variable_params}
            all_params = (TRANSLATE.to_short(ModelOptions.DEFAULT_MODEL_PARAMS)
                          | TRANSLATE.to_short(all_params))

            return all_params

        def _get_voivodeship_extra_path() -> str:
            """Return 'VOIVODESHIPS/voivodeship_name/'
            if N simulated matches any N real"""

            voivodeship_extra_path = ''

            if include_voivodeship:
                all_params = _make_all_params_shorten_dict()
                N = all_params[TRANSLATE.to_short('N')]

                if isinstance(N, int):  # if N not comes from NStabilityTester
                    # If N simulated matches N from real data,
                    # set folder name as voivodeship name.
                    real_general_data = RealData.get_real_general_data()
                    if np.sum(real_general_data['N MODEL'] == N) == 1:
                        voivodeship = real_general_data.index[
                            real_general_data['N MODEL'] == N].tolist()[0]
                        voivodeship_extra_path = f'VOIVODESHIPS/{voivodeship.capitalize()}/'

            return voivodeship_extra_path

        def _create_dir_name() -> str:
            all_params = _make_all_params_shorten_dict()

            valid_base_params = set(base_params) - set(variable_params)
            base_params_sorted = sorted(TRANSLATE.to_short(valid_base_params))

            # Append info about param values to dir like
            # 'Runs=10__g_size=(10, 10)...'
            dir_name = (Directories.AVG_SAVE_DIR
                        + _get_voivodeship_extra_path()
                        + 'raw data/'
                        + f'Runs={runs}__')

            for param in base_params_sorted:
                val = all_params[param]
                try:
                    dir_name += f'{param}={str(round(val, 3))}__'
                except TypeError:
                    print(f"AVG RESULTS {val = }")
                    tuple_val = tuple(round(x, 3) for x in val)
                    dir_name += f'{param}={tuple_val}__'

            return f'{dir_name[: -2]}/'

        def _get_file_id() -> int:
            try:
                file_id = len(
                    [path for path in Path(f"./{save_dir}").iterdir() if
                     path.is_file()])
            except FileNotFoundError:
                file_id = 0

            return file_id

        def _create_fname(_tuple_key) -> str:
            """Create fname from variable params and tuple key."""

            file_id = _get_file_id()
            fname = f'Id={str(file_id).zfill(4)}'

            for param, val in zip(TRANSLATE.to_short(variable_params),
                                  _tuple_key):
                # param vals are only floats or tuples of floats
                try:
                    fname += f'__{param}=' + f"{val:.3f}"
                except TypeError:
                    _tuple_val = tuple(round(x, 3) for x in val)
                    fname += f'__{param}={_tuple_val}'

            return f'{fname}.csv'

        save_dir = _create_dir_name()
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Saving dataframes
        for tuple_key, df in avg_results.items():
            filename = _create_fname(_tuple_key=tuple_key)
            df.to_csv(path_or_buf=save_dir + filename, index=False)

        return save_dir

    @classmethod
    def remove_tmp_results(cls):
        try:
            shutil.rmtree(Directories.TMP_SAVE_DIR)
        except OSError as e:
            print(f"Error: {Directories.TMP_SAVE_DIR} : {e.strerror}")
