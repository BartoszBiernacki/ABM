import pandas as pd
import shutil

from my_math_utils import *
from my_models.mesa_grid_star.real_data import RealData
from text_processing import *


def get_single_results(not_avg_data_directory='TMP_SAVE/'):
    """
    Returns dict in which:
        keys are tuples of variable_params and run
        values are result dataframes (one dataframe = one run of simulation)

    Intended to plot stochastic of simulations.
    """
    fnames = all_fnames_from_dir(directory=not_avg_data_directory)
    unique_tuples = get_list_of_tuples_from_dir(directory=not_avg_data_directory)
    
    data_collector_single_results = {}
    for i, fname in enumerate(fnames):
        data_collector_single_results[unique_tuples[i]] = pd.read_pickle(fname)
    
    return data_collector_single_results


def get_avg_results(directory, variable_params, ignore_dead_pandemics=False):
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
    
    data_collector_model_results = get_single_results(not_avg_data_directory=directory)
    
    num_of_variable_model_params = len(variable_params)
    list_of_tuples = get_list_of_tuples_from_dir(directory=directory)
    tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples, start_length=num_of_variable_model_params)
    
    result = {}
    for key in tuples_grouped.keys():  # key is a tuple by which other tuples were grouped. For example key=(5, 2, 2)
        lis = []
        for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
            df = data_collector_model_results[item]
            if ignore_dead_pandemics:
                last_idx = len(df.index) - 1
                
                incubation_people = df['Incubation people'][last_idx]
                prodromal_people = df['Prodromal people'][last_idx]
                illness_people = df['Illness people'][last_idx]
                incubation_cashiers = df['Incubation cashiers'][last_idx]
                prodromal_cashiers = df['Prodromal cashiers'][last_idx]
                
                if incubation_people + prodromal_people + illness_people +\
                        incubation_cashiers + prodromal_cashiers != 0:
                    lis.append(df)  # list of results dataframes matching key_tuple=(5, 2, 2)
            else:
                lis.append(df)  # list of results dataframes matching key_tuple=(5, 2, 2)
        
        if lis:
            array_with_all_iterations_results_for_specific_parameters = np.array(lis)
            
            average_array = np.mean(array_with_all_iterations_results_for_specific_parameters, axis=0)
            df = pd.DataFrame(data=average_array)
            df.columns = data_collector_model_results[list_of_tuples[0]].columns
            
            result[key] = df

    return result


def save_avg_results(avg_results,
                     fixed_params,
                     variable_params,
                     save_dir,
                     runs,
                     base_params=None,
                     include_voivodeship=True):
    """
    Saves averaged simulation results to files.
    
    :param avg_results: dict containing averaged data to save, details:
        key is a tuple containing values of variable params by which data were averaged.
        vale is a DataFrame containing averaged data from many simulations which were started with the
        variable parameters as in key.
    :type avg_results: dictionary
    :param fixed_params: fixed_params passed to run the model.
    :type fixed_params: dictionary
    :param variable_params: variable_params passed to run the model.
    :type variable_params: dictionary
    :param save_dir: directory where will be saved.
    :type save_dir: string
    :param runs: number of simulation iterations for given set of parameters.
    :type runs: integer.
    :param base_params: names of model parameters which values will be included in resulting folder name.
    :type base_params: list
    :param include_voivodeship: include voivodeship in folder name if number of households in simulation matches
        number of households in some voivodeship?
    :type include_voivodeship: boolean
    
    :return: directory to saved data
    :rtype: string
    """

    all_params = {**fixed_params, **variable_params}
    
    if include_voivodeship:
        if 'N' in all_params.keys():
            N = all_params['N']
    
            real_data_obj = RealData(customers_in_household=3)
            real_general_data = real_data_obj.get_real_general_data()
            
            # if N simulated matches N from real data, named folder as voivodeship name
            if np.sum(real_general_data['N MODEL'] == N) == 1:
                voivodeship = real_general_data.index[real_general_data['N MODEL'] == N].tolist()[0]
                save_dir += voivodeship.capitalize() + '/'
            
    if base_params:
        dir_name = f'Runs={runs}___'
        for param, val in all_params.items():
            if param in base_params:
                try:
                    dir_name += param.capitalize() + '=' + str(round(val, 3)) + '___'
                except TypeError:
                    tuple_val = tuple([round(x, 3) for x in val])
                    dir_name += param.capitalize() + '=' + str(tuple_val) + '___'
        
        dir_name = dir_name[: -3]
        save_dir += dir_name + '/'
        save_dir += 'raw data/'
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        file_id = len([path for path in Path("./" + save_dir).iterdir() if path.is_file()])
    except FileNotFoundError:
        file_id = 0
    
    # Saving dataframes
    for tuple_key, df in avg_results.items():
        fname = f'Id={str(file_id).zfill(4)}'
        for param, val in zip(variable_params, tuple_key):
            try:
                fname += '___' + param.capitalize() + '=' + str(round(val, 4))
            except TypeError:
                tuple_val = tuple([round(x, 3) for x in val])
                fname += '___' + param.capitalize() + '=' + str(tuple_val)
        
        df.to_csv(path_or_buf=save_dir + fname + '.csv', index=False)
        file_id += 1
        
    return save_dir
        
        
def remove_tmp_results():
    dir_path = 'TMP_SAVE/'
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
