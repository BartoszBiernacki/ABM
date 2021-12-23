import cProfile
import pstats

from mesa.batchrunner import BatchRunner
from mesa_batchrunner_modified import BatchRunnerMP     # saves results on disk to keep RAM free

from disease_model import DiseaseModel
from avg_results import *
from text_processing import *


def run_background_simulation(variable_params, fixed_params, multi, profiling, iterations, max_steps):
    
    fixed_params['max_steps'] = max_steps
    if multi:
        if profiling:
            with cProfile.Profile() as pr:
                batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                          nr_processes=os.cpu_count(),
                                          variable_parameters=variable_params,
                                          fixed_parameters=fixed_params,
                                          iterations=iterations,
                                          max_steps=max_steps)
                batch_run.run_all()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(10)
        else:
            batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                      nr_processes=os.cpu_count(),
                                      variable_parameters=variable_params,
                                      fixed_parameters=fixed_params,
                                      iterations=iterations,
                                      max_steps=max_steps)
            batch_run.run_all()
    else:
        if profiling:
            with cProfile.Profile() as pr:
                batch_run = BatchRunner(model_cls=DiseaseModel,
                                        variable_parameters=variable_params,
                                        fixed_parameters=fixed_params,
                                        iterations=iterations,
                                        max_steps=max_steps)
                batch_run.run_all()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(10)
        else:
            batch_run = BatchRunner(model_cls=DiseaseModel,
                                    variable_parameters=variable_params,
                                    fixed_parameters=fixed_params,
                                    iterations=iterations,
                                    max_steps=max_steps)
            batch_run.run_all()
            

def run_and_save_simulations(variable_params, fixed_params, save_dir,
                             iterations, max_steps,
                             base_params=None):
    
    run_background_simulation(variable_params=variable_params,
                              fixed_params=fixed_params,
                              multi=True,
                              profiling=False,
                              iterations=iterations,
                              max_steps=max_steps)
    
    results = get_avg_results(directory='TMP_SAVE/', variable_params=variable_params)

    save_avg_results(avg_results=results,
                     fixed_params=fixed_params,
                     variable_params=variable_params,
                     save_dir=save_dir,
                     runs=iterations,
                     base_params=base_params)

    remove_tmp_results()


def get_days_and_deaths_by_beta_and_fixed_mortality_from_dir(directory, const_mortality):
    fnames = all_fnames_from_dir(directory=directory)
    
    result = {}
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        mortality = float(variable_params['mortality'])
        if mortality == const_mortality:
            beta = variable_params['$\\beta$']
            result[beta] = df[['Day', 'Dead people']].copy()
        
    if not result:
        raise ValueError(f'Not found any data for const mortality = {const_mortality} in {directory}')
        
    return result


def get_days_and_deaths_by_mortality_and_fixed_beta_from_dir(directory, const_beta):
    fnames = all_fnames_from_dir(directory=directory)
    
    result = {}
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        beta = float(variable_params['$\\beta$'])
        if beta == const_beta:
            mortality = variable_params['mortality']
            result[mortality] = df[['Day', 'Dead people']].copy()
    
    if not result:
        raise ValueError(f'Not found any data for const mortality = {const_beta} in {directory}')
    
    return result
