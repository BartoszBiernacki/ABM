import cProfile
import copy
import pstats
import time
from typing import Union, Optional

from mesa.batchrunner import BatchRunner
# Save results on disk to keep RAM free
from disease_spread_model.mesa_modified.mesa_batchrunner_modified import BatchRunnerMP

from disease_spread_model.model.disease_model import DiseaseModel
from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.config import ModelOptions
from disease_spread_model.config import Directories
from disease_spread_model.config import HInfProb


def get_h_inf_prob_proper_value(h_inf_prob: Union[HInfProb, float],
                                beta: float) -> float:
    if isinstance(h_inf_prob, (float, int)):
        h_inf_prob = float(h_inf_prob)
        if 0 <= h_inf_prob <= 1:
            return h_inf_prob

    elif h_inf_prob == HInfProb.DEFAULT:
        return ModelOptions.HOUSEMATE_INFECTION_PROBABILITY

    elif h_inf_prob == HInfProb.BY_BETA:
        num_of_days = ModelOptions.AVG_PRODROMAL_PERIOD + ModelOptions.AVG_ILLNESS_PERIOD

        return 1 - (1 - beta) ** num_of_days

    else:
        raise ValueError(f'To get proper `h_inf_prob` pass `HInfProb` ENUM or float'
                         f'Was passed {h_inf_prob}')


class Logger:
    """Use `log(run_sim_all_params, avg_dir)` method
    to log run details to csv."""

    csv_fdir = Directories.LOGGING_MODEL_RUNS_FNAME

    @classmethod
    def _get_all_logs(cls) -> pd.DataFrame:
        try:
            df = pd.read_csv(cls.csv_fdir)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        return df

    @classmethod
    def _merge_df_from_csv_with_resulting_df(
            cls, one_row_log_df: pd.DataFrame) -> pd.DataFrame:

        all_logs_df = cls._get_all_logs()  # get df from file
        one_row_log_df['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        return pd.concat([all_logs_df, one_row_log_df])

    @staticmethod
    def _format_data_in_df(df: pd.DataFrame) -> None:
        """Convert floats to nicely formatted strings.
         `k` in suffix `_kf` stands for float precision."""

        cols_2f = ['visibility']
        cols_4f = ['mortality', 'beta', 'housemate_infection_probability']
        cols_7f = ['percent_of_infected_customers_at_start']

        for col in cols_2f:
            df[col] = [f'{val:.2f}' for val in df[col]]

        for col in cols_4f:
            df[col] = [f'{val:.4f}' for val in df[col]]

        for col in cols_7f:
            df[col] = [f'{val:.7f}' for val in df[col]]

    @classmethod
    def _overwrite_csv(cls, df: pd.DataFrame) -> None:
        """Save new df to csv."""

        save_dir = os.path.split(cls.csv_fdir)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        df.sort_values(by='run_purpose', inplace=True)
        df.to_csv(cls.csv_fdir, index=False)

    @classmethod
    def _params_to_df(
            cls, run_sim_all_params: dict, avg_dir: str) -> pd.DataFrame:
        """Convert kwargs passed to run simulation to pd.DataFrame."""

        vals_to_save = {}
        for k, v in run_sim_all_params.items():
            if isinstance(v, list):
                k, v = k, *v    # Params in `sweep_params` are lists with one elem, so unpack them.
            vals_to_save[k] = [v]

        vals_to_save['avg fname'] = avg_dir
        return pd.DataFrame(data=vals_to_save)

    @classmethod
    def log(cls, run_sim_all_params: dict, fdir: str) -> None:

        one_row_log_df = cls._params_to_df(run_sim_all_params, fdir)
        merged_logs_df = cls._merge_df_from_csv_with_resulting_df(one_row_log_df)
        cls._format_data_in_df(merged_logs_df)
        cls._overwrite_csv(merged_logs_df)


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
            variable_params: dict,
            fixed_params: dict,
            iterations: int,
            max_steps: int,
            remove_single_results=False,
            base_params=('grid_size', 'N', 'customers_in_household'),
    ):
        # Avoid circular import
        from disease_spread_model.data_processing.avg_results import Results

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

        avg_results = Results.get_avg_results(variable_params=variable_params,
                                              ignore_dead_pandemics=True)

        directory = Results.save_avg_results(avg_results=avg_results,
                                             fixed_params=fixed_params,
                                             variable_params=variable_params,
                                             runs=iterations,
                                             base_params=base_params)

        if remove_single_results:
            Results.remove_tmp_results()

        return directory

    @staticmethod
    def _all_combinations_of_variable_params(
            variable_params: dict[str, list]) -> list[dict[str, list]]:
        """
        Returns a list of dictionaries, where each dict is a cartesian
        product of values in lists in original dict. Keys to result dict are
        the same as in original dict. Vals of result dict are list with only
        one elem (they are lists to be compatible with how the model works.).

        Example:
            `variable_params` = {'a': [1], 'b': ['rower', 'łódź']}

            --> [{'a': [1], 'b': ['rower']},
                {'a': [1], 'b': ['łódź']}]
        """

        result = []
        for tuple_product in itertools.product(*variable_params.values()):
            product_dict = {k: [v] for k, v in zip(variable_params, tuple_product)}
            result.append(product_dict)

        return result

    @classmethod
    def run_simulation_to_test_sth(
            cls,
            make_log: bool,
            sweep_params: dict[str: list[float]],
            run_purpose: Optional[str] = 'None',

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
            housemate_infection_probability=HInfProb.BY_BETA,
            max_steps=ModelOptions.MAX_STEPS,

            iterations=12,
    ):
        """
        Runs simulation in separate packets (for loop), for each combination of variable parameters,
        so after each for iteration appears new file in `avg_dir` so it is easy to log all runs
        while keeping `fdir`.
         """

        # Get dict of all args, needed to initialize Model obj,
        # passed to this function.
        passed_args = locals()
        passed_args.pop('cls')
        passed_args.pop('make_log')
        passed_args['housemate_infection_probability'] = get_h_inf_prob_proper_value(
            h_inf_prob=housemate_infection_probability, beta=beta)

        variable_params = passed_args.pop('sweep_params')
        run_details = copy.deepcopy(passed_args)

        passed_args.pop('iterations')
        passed_args.pop('run_purpose')

        fixed_params = {k: passed_args[k] for k in
                        set(passed_args) - set(variable_params)}

        # run simulation --------------------------------------
        for variable_params_dict_vector in cls._all_combinations_of_variable_params(variable_params):
            avg_dir = RunModel.run_and_save_simulations(
                variable_params=variable_params_dict_vector,
                fixed_params=fixed_params,
                iterations=iterations,
                max_steps=max_steps,
                remove_single_results=True,
            )

            # Log run
            if make_log:
                # Add `variable_params` as single entries, not as dict in dict.
                run_details = run_details | variable_params_dict_vector
                fdir = latest_file_in_dir(avg_dir)
                Logger.log(run_details, fdir)


if __name__ == '__main__':
    RunModel.run_simulation_to_test_sth(
        make_log=False,
        sweep_params={
            'mortality': [1 / 100, 1.5 / 100],
            'beta': [2 / 100, 3 / 100],
                      },
        iterations=3
    )
