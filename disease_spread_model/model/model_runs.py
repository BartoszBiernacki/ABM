import cProfile
import pstats

from mesa.batchrunner import BatchRunner
# saves results on disk to keep RAM free
from disease_spread_model.mesa_modified.mesa_batchrunner_modified import BatchRunnerMP

from disease_spread_model.model.disease_model import DiseaseModel
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.config import Config


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
    def run_and_save_simulations(
            cls,
            variable_params,
            fixed_params,
            iterations,
            max_steps,
            remove_single_results=False,
            base_params=(
                    'grid_size',
                    'N',
                    'customers_in_household',
                    'infected_cashiers_at_start',
            ),
    ):
        
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
    
    @classmethod
    def run_simulation_with_one_sweep(
            cls,
            sweep_params: dict[str: list[float]],
          
            grid_size=(20, 20),
            N=700,
            customers_in_household=3,
            beta=0.025,
            mortality=0.02,
            visibility=0.65,
            
            avg_incubation_period=Config.avg_incubation_period,
            incubation_period_bins=Config.incubation_period_bins,
            avg_prodromal_period=Config.avg_prodromal_period,
            prodromal_period_bins=Config.prodromal_period_bins,
            avg_illness_period=Config.avg_illness_period,
            illness_period_bins=Config.illness_period_bins,
            
            infected_cashiers_at_start=20,
            percent_of_infected_customers_at_start=5,
            extra_shopping_boolean=True,
            housemate_infection_probability=0.1,
            max_steps=250,
            
            iterations=12,
    ):
        
        # get dict of all args, needed to initialize Model obj, passed to this function
        passed_args = locals()
        passed_args.pop('cls', None)
        passed_args.pop('iterations', None)
        
        # get dict of sweep params
        variable_params = passed_args.pop('sweep_params')
        
        # create dict of fixed params
        fixed_params = {k: passed_args[k] for k in set(passed_args) - set(variable_params)}
        
        # run simulation --------------------------------------
        res = RunModel.run_and_save_simulations(
            variable_params=variable_params,
            fixed_params=fixed_params,
            iterations=iterations,
            max_steps=max_steps,
            remove_single_results=True,
        )


if __name__ == '__main__':
    run = True
    
    param_range1 = np.linspace(0.01, .15, 5)
    # param_range2 = np.logspace(np.log10(0.3), np.log10(30), 7)
    param_range2 = np.logspace(np.log10(1), np.log10(30), 5)
    
    param_range2 = np.round(param_range2, decimals=2)
    param_range1 = np.round(param_range1, decimals=2)
    
    if run:
        RunModel.run_simulation_with_one_sweep(
            sweep_params={
                'percent_of_infected_customers_at_start': param_range2,
                # 'housemate_infection_probability': param_range1,
                'beta': [2.5 / 100],
                'mortality': [2 / 100],
                'visibility': [0.65],
            },
    
            grid_size=(20, 20),
            N=700,
            customers_in_household=3,
            beta=0.025,
            mortality=0.02,
            visibility=0.65,
    
            avg_incubation_period=Config.avg_incubation_period,
            incubation_period_bins=Config.incubation_period_bins,
            avg_prodromal_period=Config.avg_prodromal_period,
            prodromal_period_bins=Config.prodromal_period_bins,
            avg_illness_period=Config.avg_illness_period,
            illness_period_bins=Config.illness_period_bins,
    
            infected_cashiers_at_start=20,
            percent_of_infected_customers_at_start=0,
            extra_shopping_boolean=True,
            housemate_infection_probability=0.1,
            max_steps=200,
    
            iterations=11,
        )
    
