import itertools
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path


def get_list_of_tuples_from_dir(directory):
    fnames = all_fnames_from_dir(directory=directory)
    
    for i, fname in enumerate(fnames):
        tup = eval(fname[len(directory): -4])
        fnames[i] = tup
    
    return fnames


# Returns dict in which keys are tuples in which entries are values of variable parameters and
# items are averaged model parameters
def group_tuples_by_start(list_of_tuples, start_length):
    result = {}
    tuples_starts = [[item for i, item in enumerate(tup) if i < start_length] for tup in list_of_tuples]
    tuples_starts.sort()
    unique_tuples_starts = list(tuples_starts for tuples_starts, _ in itertools.groupby(tuples_starts))

    for unique_tuple_start in unique_tuples_starts:
        tuples_grouped_by_start = []
        for tup in list_of_tuples:
            tuple_start = tup[:start_length]
            if list(tuple_start) == unique_tuple_start:
                tuples_grouped_by_start.append(tup)
        result[tuple(unique_tuple_start)] = tuples_grouped_by_start
    return result


def variable_params_from_fname(fname):
    # fname = 'blah blah bla/.../results/Grid_size=(1, 1)/Id=0___N=700___Beta_mortality_pair=(0.03, 0).csv'
    
    new = fname
    while '/' in new:
        new = new[new.find('/') + 1:]
    new = new[: new.find('.csv')]
    
    new_list = new.split(sep='___')
    new_new_list = []
    for i, param in enumerate(new_list):
        if 'Id=' in param:
            pass
        
        elif 'Beta_mortality_pair=' in param:
            beta = float(param[param.find('=(') + 2: param.find(', ')])
            mortality = float(param[param.find(', ') + 2: param.find(')')])
            
            new_new_list.append(f'beta={beta}')
            new_new_list.append(f'mortality={mortality}')
        
        elif 'Grid_size=' in param:
            new_new_list.append(new_list[i].replace("Grid_size=", "Grid size="))
        
        elif 'Infect_housemates_boolean=' in param:
            new_new_list.append(new_list[i].replace("Infect_housemates_boolean=", "Infect housemates="))
        
        else:
            new_new_list.append(new_list[i])
    
    my_dict = {}
    for param in new_new_list:
        key = param[: param.find('=')]
        if key == 'beta':
            key = r'$\beta$'
        elif key == 'N':
            pass
        else:
            key = key.lower()
        val = param[param.find('=') + 1:]
        my_dict[key] = val
        
        # Sometimes using 'beta' is more convenient than r'$\beta$' so add them both.
        if key == r'$\beta$':
            my_dict['beta'] = val
            
    return my_dict


def fixed_params_from_fname(fname):
    """
    Gets path to avg result data file and returns fixed params for which simulation was run.
    
    :param fname: path to the averaged simulation results
    :type fname: str
    :return: fixed params of simulation which produced that result. Result is a dictionary in which:
        key: name of parameter
        val: value of that parameter
    :rtype: dict
    """
    
    # from file path get folder name which describes fixed params of saved simulation result
    new = fname[fname.find('Runs'):]
    new = new[:new.find('/')]
    
    new_list = new.split(sep='___')
    new_new_list = []
    for i, param in enumerate(new_list):
        if 'Beta_mortality_pair=' in param:
            beta = float(param[param.find('=(') + 2: param.find(', ')])
            mortality = float(param[param.find(', ') + 2: param.find(')')])
            
            new_new_list.append(f'beta={beta}')
            new_new_list.append(f'mortality={mortality}')
        
        elif 'Num_of_infected_cashiers_at_start=' in param:
            infected_at_start = int(param[param.find('=') + 1:])
            new_new_list.append(f'Infected cashiers at start={infected_at_start}')
        
        elif 'Infect_housemates_boolean=' in param:
            new_new_list.append(new_list[i].replace("Infect_housemates_boolean=", "Infect housemates="))
        
        else:
            new_new_list.append(new_list[i].replace('_', ' '))
    
    my_dict = {}
    for i, param in enumerate(new_new_list):
        key = param[: param.find('=')]
        if key == 'Grid_size':
            key = 'Grid size'
        val = param[param.find('=') + 1:]
        my_dict[key] = val
    
    return my_dict


def voivodeship_from_fname(fname):
    """
    If it can it returns voivodeship from fname, otherwise returns None.
    
    :param fname: path to file from which voivodeship will be recognized.
    :type fname: str
    """

    # needs to be imported here, otherwise circular import error
    from real_data import RealData
    
    # infinite loop protection
    i = 0
    while len(fname) > 0:
        folder_name = fname[: fname.find('/')]
        fname = fname[fname.find('/')+1:]
        
        if folder_name.lower() in RealData.get_voivodeships():
            return folder_name.lower()
        
        if '/' not in fname:
            return None
        
        if i > 50:
            raise IndexError('voivodeship_from_fname stuck in infinity loop!')


def all_fnames_from_dir(directory):
    fnames = []
    cwd = os.getcwd()
    Path(directory).mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    for file in glob.glob("*.*"):
        fnames.append(directory + file)
    os.chdir(cwd)
    
    fnames.sort()
    
    return fnames


def get_ax_title_from_fixed_params(fixed_params):
    ax_title = ''
    i = 0
    for key, val in fixed_params.items():
        if key == 'Infect housemates':
            val = str(bool(int(val)))
        ax_title += key + '=' + val + ' ' * 4
        i += 1
        if i % 3 == 0:
            ax_title = ax_title[:-4]
            ax_title += '\n'
    
    return ax_title


def get_legend_from_variable_params(variable_params, ignored_params):
    
    # variable_params has {'beta': val} and also {r'$\beta$': val}
    # for legend purpose use only r'$\beta$'
    ignored_params.append('beta')
    legend = ''
    i = 0
    for key, val in variable_params.items():
        if key not in ignored_params:
            if key == 'mortality':
                legend += key + '=' + str(round(float(val) * 100, 2)) + '%' + ' ' * 4
            else:
                legend += key + '=' + val + ' ' * 4
            i += 1
            if i % 3 == 0:
                ax_title = legend[:-4]
                ax_title += '\n'
    
    return legend


def dict_to_fname_str(dictionary):
    result = ''
    for key, val in dictionary.items():
        result += str(key) + '=' + str(val) + '___'
    
    return result[:-3]


# For ,,Death toll'' plots *************************************************************************************


def group_fnames_by_pair_param2_param3_and_param1(directory: str,
                                                  param1: str,
                                                  param2: str,
                                                  param3: str):
    """

    :param directory: directory do folder which contains avg resulting files
    :type directory: str
    :param param1: name of model parameter, available ['beta', 'mortality', 'visibility']
    :type param1: str
    :param param2: like param1, but different than param1
    :type param2: str
    :param param3: like param1, but different than param1 and param2
    :type param3: str
    :return: 2D list of fnames where:
        result[i][j] --> fname( param1_i, (param2, param3)_j )
    :rtype: np.ndarray
    
    Example:
        param1, param2, param3 = beta, mortality, visibility
        result[i][j] --> fname( (mortality, visibility)_i, beta_j )
    """

    # prepare dict which will contain unique values of param1
    params1_vals_to_index = {}
    # prepare dict which will contain unique values of pairs (param2, param3)
    pair_params2_params3_vals_to_index = {}
    
    # fnames to group
    fnames = all_fnames_from_dir(directory=directory)
    
    # iterate over fnames to find out all possible param1 values
    # and pairs (param2 value, param3 value)
    for fname in fnames:
        
        # read variable_params from fname
        variable_params = variable_params_from_fname(fname=fname)

        # make pair (param2 value, param3 value) add it to a dict if it appears for the first time
        tup = (variable_params[param2], variable_params[param3])
        if tup not in pair_params2_params3_vals_to_index.keys():
            pair_params2_params3_vals_to_index[tup] = len(pair_params2_params3_vals_to_index)
        
        # get value of param1 and add to a dict if it appears for the first time
        param1_value = variable_params[param1]
        if param1_value not in params1_vals_to_index.keys():
            params1_vals_to_index[param1_value] = len(params1_vals_to_index)

    # sort dict by it's values.
    # order of items in it matters while plotting death toll (line color depends on it) ---------------
    params1_vals_to_index = \
        {k: v for k, v in zip(sorted(params1_vals_to_index.keys()), range(len(params1_vals_to_index)))}
        
    # create empty resulting list
    result = np.zeros((len(pair_params2_params3_vals_to_index), len(params1_vals_to_index)), dtype=object)
    
    # iterate over fnames again and fill resulting list
    for fname in fnames:
        
        # read variable_params from fname
        variable_params = variable_params_from_fname(fname=fname)
        
        # get pair (param2 value, param3 value) and its corresponding index
        tup = (variable_params[param2], variable_params[param3])
        first_index = pair_params2_params3_vals_to_index[tup]

        # get value of param1 and its corresponding index
        param1_value = variable_params[param1]
        second_index = params1_vals_to_index[param1_value]
        
        # fill resulting list wit current fname
        result[first_index][second_index] = fname
    
    return result


def group_fnames_by_param1_param2_param3(directory: str,
                                         param1: str,
                                         param2: str,
                                         param3: str):
    """
    Groups fnames in directory with respect to simulation params that are
    included in them. Groups in order as params were given,
    Simulation params are: [visibility, beta, mortality].
    
    Result[i][j][k] --> fname(param_i, param_j, param_k)
    For example: if params are: visibility, beta, mortality:
        result[i][j][k] --> fname(param_i, param_j, param_k)
        
    In another words:
        If in directory are avg results from simulation in which was tested:
             3 beta values, 2 mortality values, 4 visibility values
        then directory contains 3*2*4=24 files. This function allows to access
        each fname by specifying [beta, mortality, visibility] unique
        number.
        
        Note: by number not value. E.g. if simulations was ran for
        betas = [0.02, 0.025, 0.03], and You want to access all fnames
        in which beta >=0.025 then use:
            param1="beta",
            param2="mortality",
            param3="visibility",
                result[1:, :, :]
                
    """
    
    # Prepare dicts for containing {paramX_occurrenceNumber: paramX_value}
    params1_vals_to_index = {}
    params2_vals_to_index = {}
    params3_vals_to_index = {}
    
    # iterate over all fnames to find out what params vales are there
    # add those values to correspondence dicts
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        
        # get visibility, beta, mortality values from fname
        variable_params = variable_params_from_fname(fname=fname)
        
        # if this is a first occurrence of param1 add it to dict (dict[value]=index)
        param1_value = variable_params[param1]
        if param1_value not in params1_vals_to_index.keys():
            params1_vals_to_index[param1_value] = len(params1_vals_to_index)
        
        # same as above
        param2_value = variable_params[param2]
        if param2_value not in params2_vals_to_index.keys():
            params2_vals_to_index[param2_value] = len(params2_vals_to_index)
        
        # same as above
        param3_value = variable_params[param3]
        if param3_value not in params3_vals_to_index.keys():
            params3_vals_to_index[param3_value] = len(params3_vals_to_index)
    
    # sort each dict by it's values.
    # order of items in them matters while plotting death toll (line color depends on it) ---------------
    params1_vals_to_index = \
        {k: v for k, v in zip(sorted(params1_vals_to_index.keys()), range(len(params1_vals_to_index)))}
    
    params2_vals_to_index = \
        {k: v for k, v in zip(sorted(params2_vals_to_index.keys()), range(len(params2_vals_to_index)))}
    
    params3_vals_to_index = \
        {k: v for k, v in zip(sorted(params3_vals_to_index.keys()), range(len(params3_vals_to_index)))}
    # ---------------------------------------------------------------------------------------------------
    
    # result will be a 3D list of fnames (strings)
    result = np.empty((len(params1_vals_to_index),
                       len(params2_vals_to_index),
                       len(params3_vals_to_index)),
                      dtype=object)
    
    # fill the result 3D list
    # iterate over all fnames again
    for fname in fnames:
        # get visibility, beta, mortality values from fname
        variable_params = variable_params_from_fname(fname=fname)
        
        # get value of param from fname
        param1_value = variable_params[param1]
        # get index of it's param value from dict that was filled in previous for loop
        # that index will be the fist index (out of 3) needed to get this file from resulting 3D list
        first_index = params1_vals_to_index[param1_value]
        
        # same as above
        param2_value = variable_params[param2]
        second_index = params2_vals_to_index[param2_value]
        
        # same as above
        param3_value = variable_params[param3]
        third_index = params3_vals_to_index[param3_value]
        
        result[first_index][second_index][third_index] = fname
    
    return result

# *****************************************************************************************************************


def rename_duplicates_in_df_index_column(df, suffix='-duplicate-'):
    appendents = (suffix + df.groupby(level=0).cumcount().astype(str).replace('0', '')).replace(suffix, '')
    return df.set_index(df.index + appendents)


def check_uniqueness_and_correctness_of_params(param1: str,
                                               param2: str,
                                               param3: str):
    """
    Raises an ValueError if passed params are not unique elements
    of list ['beta', 'mortality', 'visibility'].
    """
    
    allowed_params = ['beta', 'mortality', 'visibility']
    
    are_params_allowed = all([param in allowed_params
                              for param in [param1, param2, param3]])
    
    are_params_unique = len([param1, param2, param3]) == len({param1, param2, param3})
    
    if not (are_params_allowed and are_params_unique):
        raise ValueError(f"Model params passed to some function "
                         f"should be three unique params from  {allowed_params}, but "
                         f"was passed: param1={param1}, param2={param2}, param3={param3}.")


def get_last_simulated_day(fname: str):
    """
    Returns last day of simulation.

    :param fname: filename containing data from which last day will be read
    :type fname: str
    :return: last day of simulation.
    :rtype: int

    """
    df = pd.read_csv(fname)
    return int(max(df['Day']))