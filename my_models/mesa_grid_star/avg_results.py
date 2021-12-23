import pandas as pd
from my_math_utils import *
from text_processing import *

from pathlib import Path
import shutil


def data_dict_from_files(directory):
    fnames = all_fnames_from_dir(directory=directory)
    unique_tuples = get_list_of_tuples_from_dir(directory=directory)
    
    result = {}
    for i, fname in enumerate(fnames):
        result[unique_tuples[i]] = pd.read_pickle(fname)
    
    return result


def get_avg_results(directory, variable_params):
    # returns dict in which keys are tuples of variable_params and values are dataframes averaged over all iterations
    num_of_variable_model_params = len(variable_params)
    list_of_tuples = get_list_of_tuples_from_dir(directory=directory)
    tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples, start_length=num_of_variable_model_params)
    data_collector_model_results = data_dict_from_files(directory=directory)
    
    result = {}
    for key in tuples_grouped.keys():  # key is a tuple by which other tuples were grouped. For example key=(5, 2, 2)
        lis = []
        for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
            lis.append(data_collector_model_results[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
        array_with_all_iterations_results_for_specific_parameters = np.array(lis)
        
        average_array = np.mean(array_with_all_iterations_results_for_specific_parameters, axis=0)
        df = pd.DataFrame(data=average_array)
        df.columns = data_collector_model_results[list_of_tuples[0]].columns
        
        result[key] = df

    return result


def save_avg_results(avg_results, fixed_params, variable_params, save_dir, runs,
                     base_params=None):
    if base_params:
        dir_name = '/' + f'Runs={runs}___'
        for param, val in {**fixed_params, **variable_params}.items():
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
                fname += '___' + param.capitalize() + '=' + str(round(val, 3))
            except TypeError:
                tuple_val = tuple([round(x, 3) for x in val])
                fname += '___' + param.capitalize() + '=' + str(tuple_val)
        
        df.to_csv(path_or_buf=save_dir + fname + '.csv', index=False)
        file_id += 1
        
        
def remove_tmp_results():
    dir_path = 'TMP_SAVE/'
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
