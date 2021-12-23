import itertools
import glob
import os
import numpy as np


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
    # fname = 'results/Grid_size=(1, 1)/Id=0___N=700___Beta_mortality_pair=(0.03, 0).csv'
    
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
    
    return my_dict


def fixed_params_from_fname(fname):
    # fname = 'results/Runs=15___Grid_size=(20, 20)___N=1000___Customers_in_household=1___Infected_cashiers_at_start=400___Infect_housemates_boolean=0/raw data/Id=0000___Beta_mortality_pair=(0.01, 0.01)___Visibility=0.0.csv'
    
    new = fname
    new = new.replace('results/', '')
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


def all_fnames_from_dir(directory):
    fnames = []
    cwd = os.getcwd()
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
def group_fnames_by_beta(directory):
    # result[i][0] --> fname( beta_0, (mortality, visibility)_i )
    mortality_visibility_to_index = {}
    betas_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params['mortality'], variable_params['visibility'])
        if tup not in mortality_visibility_to_index.keys():
            mortality_visibility_to_index[tup] = len(mortality_visibility_to_index)
        
        beta = variable_params[r'$\beta$']
        if beta not in betas_to_index.keys():
            betas_to_index[beta] = len(betas_to_index)
    
    result = np.empty((len(mortality_visibility_to_index), len(betas_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params['mortality'], variable_params['visibility'])
        first_index = mortality_visibility_to_index[tup]
        
        beta = variable_params[r'$\beta$']
        second_index = betas_to_index[beta]
        
        result[first_index][second_index] = fname
    
    return result


def group_fnames_by_mortality(directory):
    # result[i][0] --> fname( mortality_0, (beta, visibility)_i )
    beta_visibility_to_index = {}
    mortalities_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params[r'$\beta$'], variable_params['visibility'])
        if tup not in beta_visibility_to_index.keys():
            beta_visibility_to_index[tup] = len(beta_visibility_to_index)
        
        mortality = variable_params['mortality']
        if mortality not in mortalities_to_index.keys():
            mortalities_to_index[mortality] = len(mortalities_to_index)
    
    result = np.empty((len(beta_visibility_to_index), len(mortalities_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params[r'$\beta$'], variable_params['visibility'])
        first_index = beta_visibility_to_index[tup]
        
        mortality = variable_params['mortality']
        second_index = mortalities_to_index[mortality]
        
        result[first_index][second_index] = fname
    
    return result


def group_fnames_by_visibility(directory):
    # result[i][0] --> fname( visibility_0, (beta, mortality)_i )
    beta_mortality_to_index = {}
    visibilities_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params[r'$\beta$'], variable_params['mortality'])
        if tup not in beta_mortality_to_index.keys():
            beta_mortality_to_index[tup] = len(beta_mortality_to_index)
        
        visibility = variable_params['visibility']
        if visibility not in visibilities_to_index.keys():
            visibilities_to_index[visibility] = len(visibilities_to_index)
    
    result = np.empty((len(beta_mortality_to_index), len(visibilities_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        tup = (variable_params[r'$\beta$'], variable_params['mortality'])
        first_index = beta_mortality_to_index[tup]
        
        visibility = variable_params['visibility']
        second_index = visibilities_to_index[visibility]
        
        result[first_index][second_index] = fname
    
    return result


def group_fnames_standard_by_mortality_beta_visibility(directory):
    # result[i][j][k] --> fname(mortality_i, beta_j, visibility_k)
    mortalities_to_index = {}
    betas_to_index = {}
    visibilities_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        
        mortality = variable_params['mortality']
        if mortality not in mortalities_to_index.keys():
            mortalities_to_index[mortality] = len(mortalities_to_index)
        
        beta = variable_params[r'$\beta$']
        if beta not in betas_to_index.keys():
            betas_to_index[beta] = len(betas_to_index)
        
        visibility = variable_params['visibility']
        if visibility not in visibilities_to_index.keys():
            visibilities_to_index[visibility] = len(visibilities_to_index)
    
    result = np.empty((len(mortalities_to_index), len(betas_to_index), len(visibilities_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        mortality = variable_params['mortality']
        first_index = mortalities_to_index[mortality]
        
        beta = variable_params[r'$\beta$']
        second_index = betas_to_index[beta]
        
        visibility = variable_params['visibility']
        third_index = visibilities_to_index[visibility]
        
        result[first_index][second_index][third_index] = fname
    
    return result


def group_fnames_standard_by_visibility_beta_mortality(directory):
    # result[i][j][k] --> fname(visibility_i, beta_j, mortality_k)
    visibilities_to_index = {}
    betas_to_index = {}
    mortalities_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)

        visibility = variable_params['visibility']
        if visibility not in visibilities_to_index.keys():
            visibilities_to_index[visibility] = len(visibilities_to_index)
        
        beta = variable_params[r'$\beta$']
        if beta not in betas_to_index.keys():
            betas_to_index[beta] = len(betas_to_index)
        
        mortality = variable_params['mortality']
        if mortality not in mortalities_to_index.keys():
            mortalities_to_index[mortality] = len(mortalities_to_index)
    
    result = np.empty((len(visibilities_to_index), len(betas_to_index), len(mortalities_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)

        visibility = variable_params['visibility']
        first_index = visibilities_to_index[visibility]

        beta = variable_params[r'$\beta$']
        second_index = betas_to_index[beta]
        
        mortality = variable_params['mortality']
        third_index = mortalities_to_index[mortality]
        
        result[first_index][second_index][third_index] = fname
    
    return result


def group_fnames_standard_by_mortality_visibility_beta(directory):
    # result[i][j][k] --> fname(mortality_i, visibility_j, beta_k)
    mortalities_to_index = {}
    visibilities_to_index = {}
    betas_to_index = {}
    
    fnames = all_fnames_from_dir(directory=directory)
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        
        mortality = variable_params['mortality']
        if mortality not in mortalities_to_index.keys():
            mortalities_to_index[mortality] = len(mortalities_to_index)

        visibility = variable_params['visibility']
        if visibility not in visibilities_to_index.keys():
            visibilities_to_index[visibility] = len(visibilities_to_index)
        
        beta = variable_params[r'$\beta$']
        if beta not in betas_to_index.keys():
            betas_to_index[beta] = len(betas_to_index)
        
    result = np.empty((len(mortalities_to_index), len(visibilities_to_index), len(betas_to_index)), dtype=object)
    
    for fname in fnames:
        variable_params = variable_params_from_fname(fname=fname)
        mortality = variable_params['mortality']
        first_index = mortalities_to_index[mortality]

        visibility = variable_params['visibility']
        second_index = visibilities_to_index[visibility]
        
        beta = variable_params[r'$\beta$']
        third_index = betas_to_index[beta]
        
        result[first_index][second_index][third_index] = fname
    
    return result
# *****************************************************************************************************************
