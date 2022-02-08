import cProfile
import pstats

from mesa.batchrunner import BatchRunner
# saves results on disk to keep RAM free
from disease_spread_model.mesa_modified.mesa_batchrunner_modified import BatchRunnerMP

from disease_spread_model.model.disease_model import DiseaseModel
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.text_processing import *


class RunModel(object):
    def __init__(self):
        pass

    @classmethod
    def _run_background_simulation(cls,
                                   variable_params,
                                   fixed_params,
                                   multi,
                                   profiling,
                                   iterations,
                                   max_steps):
        
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
            
    @classmethod
    def run_and_save_simulations(cls,
                                 variable_params,
                                 fixed_params,
                                 iterations,
                                 max_steps,
                                 remove_single_results=False,
                                 base_params=None):
    
        cls._run_background_simulation(variable_params=variable_params,
                                       fixed_params=fixed_params,
                                       multi=True,
                                       profiling=False,
                                       iterations=iterations,
                                       max_steps=max_steps)
    
        results = Results.get_avg_results(variable_params=variable_params,
                                          ignore_dead_pandemics=True)
    
        directory = Results.save_avg_results(avg_results=results,
                                             fixed_params=fixed_params,
                                             variable_params=variable_params,
                                             runs=iterations,
                                             base_params=base_params)
    
        if remove_single_results:
            Results.remove_tmp_results()
        
        return directory
