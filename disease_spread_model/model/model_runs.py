import cProfile
import copy
import pstats

from mesa.batchrunner import BatchRunner
# saves results on disk to keep RAM free
from disease_spread_model.mesa_modified.mesa_batchrunner_modified import BatchRunnerMP

from disease_spread_model.model.disease_model import DiseaseModel
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.config import ModelOptions


class RunModel:
    @classmethod
    def _get_proper_fixed_params(cls, variable_params: dict, fixed_params: dict) -> dict:
        
        # Find out which params are missing
        missing_params = set(ModelOptions.DEFAULT_MODEL_PARAMS) - set(fixed_params | variable_params)
        
        # Do not modify dict passed to function
        fixed_params = copy.copy(fixed_params)
        
        # Fill missing entries in fixed params
        for key in missing_params:
            fixed_params[key] = ModelOptions.DEFAULT_MODEL_PARAMS[key]
        
        return fixed_params
    
    @classmethod
    def _run_background_simulation(cls,
                                   variable_params,
                                   fixed_params,
                                   multi,
                                   profiling,
                                   iterations,
                                   max_steps):
        
        fixed_params = cls._get_proper_fixed_params(variable_params=variable_params,
                                                    fixed_params=fixed_params)
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
        elif profiling:
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
    def _are_base_params_valid(cls, base_params: tuple or list) -> bool:
        return all(param in ModelOptions.DEFAULT_MODEL_PARAMS for param in base_params)
    
    @classmethod
    def run_and_save_simulations(
            cls,
            variable_params,
            fixed_params,
            iterations,
            max_steps,
            remove_single_results=False,
            base_params=('grid_size', 'N', 'customers_in_household'),
    ):
        
        if not cls._are_base_params_valid(base_params):
            raise ValueError(f"'base_params' wrong because "
                             f"{set(base_params) - set(ModelOptions.DEFAULT_MODEL_PARAMS)} "
                             f"is not in 'DiseaseModel' constructor!")
        
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
            
            grid_size=ModelOptions.GRID_SIZE,
            N=ModelOptions.N,
            customers_in_household=ModelOptions.CUSTOMERS_IN_HOUSEHOLD,
            beta=ModelOptions.BETA,
            mortality=ModelOptions.MORTALITY,
            visibility=ModelOptions.VISIBILITY,
            
            avg_incubation_period=ModelOptions.AVG_INCUBATION_PERIOD,
            incubation_period_bins=ModelOptions.INCUBATION_PERIOD_BINS,
            avg_prodromal_period=ModelOptions.AVG_PRODROMAL_PERIOD,
            prodromal_period_bins=ModelOptions.PRODROMAL_PERIOD_BINS,
            avg_illness_period=ModelOptions.AVG_ILLNESS_PERIOD,
            illness_period_bins=ModelOptions.ILLNESS_PERIOD_BINS,
            
            infected_cashiers_at_start=ModelOptions.INFECTED_CASHIERS_AT_START,
            percent_of_infected_customers_at_start=ModelOptions.PERCENT_OF_INFECTED_CUSTOMERS_AT_START,
            extra_shopping_boolean=ModelOptions.EXTRA_SHOPPING_BOOLEAN,
            housemate_infection_probability=ModelOptions.HOUSEMATE_INFECTION_PROBABILITY,
            max_steps=ModelOptions.MAX_STEPS,
            
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
    param_range2 = np.logspace(np.log10(1), np.log10(30), 2)
    
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
            
            infected_cashiers_at_start=20,
            percent_of_infected_customers_at_start=0,
            housemate_infection_probability=0.1,
            
            extra_shopping_boolean=True,
            max_steps=200,
            
            iterations=2,
        )
