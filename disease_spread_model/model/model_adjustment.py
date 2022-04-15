from abc import ABC, abstractmethod
from typing import Union
from enum import Enum

from scipy.optimize import fmin_cobyla
import time

from disease_spread_model.config import Directories, ModelOptions, StartingDayBy
from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.model.model_runs import RunModel, HInfProb
from disease_spread_model.data_processing.real_data import RealData


class OptimizeParam(Enum):
    BETA = 'beta'
    MORTALITY = 'mortality'
    BETA_AND_MORTALITY = 'beta and mortality'
    NONE = 'None'

    # Constraints, assumed that beta and mortality are in percentage.
    # Constraints as inequalities 'expression' > 0, example:
    # 'lambda vec: vec[0] - 1' --> vec[0] - 1 > 0 --> beta > 1%
    BETA_CONS = [
        lambda vec: vec[0] - 1 / 100,
        lambda vec: -(vec[0] - 6 / 100),
    ]

    MORTALITY_CONS = [
        lambda vec: vec[0] - 1 / 100,
        lambda vec: -(vec[0] - 4 / 100),
    ]

    BETA_AND_MORTALITY_CONS = [
        lambda vec: vec[0] - 1 / 100,
        lambda vec: -(vec[0] - 6 / 100),
        lambda vec: vec[1] - 1 / 100,
        lambda vec: -(vec[1] - 4 / 100),
    ]


class OptimizeResultsWriterReader:
    """Use `to_csv(df)` method to save tuning results."""

    @staticmethod
    def get_all_tuning_results() -> pd.DataFrame:
        try:
            df = pd.read_csv(Directories.TUNING_MODEL_PARAMS_FNAME)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        return df

    @classmethod
    def get_n_best_tuned_results(cls, n: int) -> pd.DataFrame:
        df = cls.get_all_tuning_results()
        result_df = pd.DataFrame(columns=df.columns)

        for voivodeship in df['voivodeship'].unique():
            result_df = pd.concat([result_df, df.loc[df['voivodeship'] == voivodeship].iloc[:n]])

        result_df.reset_index(inplace=True, drop=True)
        return result_df

    @classmethod
    def _merge_df_from_csv_with_resulting_df(cls, df_tuning_details: pd.DataFrame) -> pd.DataFrame:
        df = cls.get_all_tuning_results()  # get df from file
        df_tuning_details['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        df = pd.concat([df, df_tuning_details])  # insert row (merge)

        # sort df
        df.sort_values(by=['voivodeship', 'fit error per day'], inplace=True, ignore_index=True)

        return df

    @staticmethod
    def _format_data_in_df(df: pd.DataFrame) -> None:
        """Convert floats to nicely formatted strings.
         `k` in suffix `_kf` stands for float precision."""

        cols_2f = ['fit error per day', 'visibility']
        cols_4f = ['mortality', 'beta', 'h inf prob']
        cols_7f = ['fraction of infected customers at start']

        for col in cols_2f:
            df[col] = [f'{val:.2f}' for val in df[col]]

        for col in cols_4f:
            df[col] = [f'{val:.4f}' for val in df[col]]

        for col in cols_7f:
            df[col] = [f'{val:.7f}' for val in df[col]]

    @classmethod
    def _overwrite_csv(cls, df: pd.DataFrame) -> None:
        """Save new df to csv."""

        save_dir = os.path.split(Directories.TUNING_MODEL_PARAMS_FNAME)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Directories.TUNING_MODEL_PARAMS_FNAME, index=False)

    @classmethod
    def to_csv(cls, df_new_tuning_details: pd.DataFrame) -> None:

        df = cls._merge_df_from_csv_with_resulting_df(df_new_tuning_details)
        cls._format_data_in_df(df)
        cls._overwrite_csv(df)


class Optimizer(ABC):

    def __init__(
            self,
            voivodeships: list[str],
            ignored_voivodeships: list[str],
            starting_days_by: StartingDayBy,
            percent_of_touched_counties: int,
            last_date_str_YYYY_mm_dd: str,

            beta: float,
            mortality: float,
            visibility: float,
            housemate_infection_probability: Union[HInfProb, float],

            max_sim_days: int,
            simulations_to_avg: int,
            num_of_shots: int,
    ):

        self.voivodeships = voivodeships
        self.ignored_voivodeships = ignored_voivodeships

        self.starting_days = RealData.starting_days(
            by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
        )
        self.ending_days = RealData.ending_days_by_death_toll_slope(
            starting_days_by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date=last_date_str_YYYY_mm_dd,
        )

        self.last_date_str_YYYY_mm_dd = last_date_str_YYYY_mm_dd

        self.beta_init = beta
        self.beta = beta
        self.mortality_init = mortality
        self.mortality = mortality
        self.visibility = visibility
        self.h_inf_prob = housemate_infection_probability

        self.fractions_of_init_infected_customers = {
            voivodeship:
                self._get_fraction_of_init_infected_customers(voivodeship)
            for voivodeship in self.voivodeships
        }

        self.max_sim_days = max_sim_days
        self.simulations_to_avg = simulations_to_avg
        self.num_of_shots = num_of_shots

    def _get_h_inf_prob_proper_value(self) -> float:
        if isinstance(self.h_inf_prob, (float, int)):
            h_inf_prob = float(self.h_inf_prob)
            if 0 <= h_inf_prob <= 1:
                return h_inf_prob

        elif self.h_inf_prob == HInfProb.DEFAULT:
            return ModelOptions.HOUSEMATE_INFECTION_PROBABILITY

        elif self.h_inf_prob == HInfProb.BY_BETA:
            num_of_days = ModelOptions.AVG_PRODROMAL_PERIOD + ModelOptions.AVG_ILLNESS_PERIOD

            return 1 - (1 - self.beta) ** num_of_days

        else:
            raise ValueError(f'To get proper `h_inf_prob` pass `HInfProb` ENUM or float'
                             f'Was passed {self.h_inf_prob}')

    def _get_fraction_of_init_infected_customers(self, voivodeship: str) -> float:
        """
        Returns an approximate fraction of infected customers in
        given voivodeship, based on RealData, which can be used to
        speed up convergence of Optimizer (reduces shift).

        Returns:
            ((deaths/mortality) / population) * 25%
        Note: 25% instead of 100% to not overshoot death toll right away and
            because infections between housemates.
        """
        death_tolls = RealData.death_tolls()
        death_toll = death_tolls.loc[voivodeship]
        date = RealData.day_to_date(day_number=self.starting_days[voivodeship] + 7)
        num_of_deaths = death_toll[date]
        num_of_infected = num_of_deaths / self.mortality

        geospatial_df = RealData.get_real_general_data()
        population = geospatial_df.loc[voivodeship]['population']

        # return 0
        return (num_of_infected / population) * .25

    @abstractmethod
    def _func_to_optimize(self, vector: np.ndarray, extra_args: dict) -> float:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @staticmethod
    def _find_best_x_shift_to_match_plots(
            y1_reference: np.ndarray, y2: np.ndarray,
            y2_start_idx: int, y2_end_idx: int) -> (int, float):
        """
        Returns index of elem from which data y2[start: stop]
        best match any slice of the same length of y1.
        Also returns fit error (SSE).

        Pure math function, no simulation.

        Example:
            y1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            y2 = [3, 5, 1, 4, 5, 6, 7, 9, 11, 5, 4]
            y2_start_idx = 3, (y2[3] = 4)
            y2_end_idx = 6, (y2[6] = 7)

            --> shift = 1, error = 0

            Shift = 1, because common part [4, 5, 6, 7] in:
                - y1 starts on idx1 = 2
                - y2 starts on idx2 = 3
            so common part on y2 is shifted to the right by 1 unit
            compared to y1 that is a reference line.

            Error = 0, because common part of y1 and y2
            overlaps perfectly.

        """
        y1_reference = np.array(y1_reference)
        y2 = np.array(y2[y2_start_idx: y2_end_idx + 1])

        smallest_difference = 1e9
        y2_length = y2_end_idx - y2_start_idx + 1
        shift = 0

        for i in range(len(y1_reference) - len(y2) + 1):
            y1_subset = y1_reference[i: i + y2_length]

            difference = np.sum((y2 - y1_subset) ** 2)

            if difference < smallest_difference:
                smallest_difference = difference
                shift = i

        shift = y2_start_idx - shift

        return shift, smallest_difference

    def _create_one_row_df_result_details(self,
                                          voivodeship: str,
                                          fit_error_per_day: float,
                                          shift: int,
                                          avg_fname: str,
                                          optimize_param: OptimizeParam,
                                          ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "voivodeship": [voivodeship],
                "fit error per day": [fit_error_per_day],
                "beta": [self.beta],
                "mortality": [self.mortality],
                "visibility": [self.visibility],
                "h inf prob": [self._get_h_inf_prob_proper_value()],

                "fraction of infected customers at start": [self.fractions_of_init_infected_customers[voivodeship]],

                'first day': [self.starting_days[voivodeship]],
                'last day': [self.ending_days[voivodeship]],
                "runs": [self.simulations_to_avg],

                'shift': [shift],
                'fname': [avg_fname],
                'optimize method': [optimize_param.value]
            },
            index=[-1],
        )

    def _find_shift_and_fit_error_per_day(self, voivodeship: str) -> (int, float, str):
        """
        A general purpose method of this class.
        Runs simulation for given params and returns best
        shift and fit error and avg_fname.

        Runs simulations to evaluate how similar real data are to simulation data.
        Similarity is measured since beginning of pandemic for given voivodeship specified
        by percent_of_touched_counties. Real data are shifted among X axis to best match
        simulated data (only in specified interval).

        Returns:
            * `shift`,
            * `fit_error` = SSE / days_to_fit_death_toll,
            * `avg_fname`.
        """

        # Get extra values needed to run simulation
        real_general_data = RealData.get_real_general_data()
        grid_side_length = real_general_data.loc[voivodeship, 'grid side MODEL']
        h_inf_prob = self._get_h_inf_prob_proper_value()

        # Run simulation
        directory = RunModel.run_and_save_simulations(
            fixed_params={
                "grid_size": (grid_side_length, grid_side_length),
                "N": real_general_data.loc[voivodeship, 'N MODEL'],
                "infected_cashiers_at_start": 10,
                "percent_of_infected_customers_at_start":
                    self.fractions_of_init_infected_customers[voivodeship] * 100,
            },
            variable_params={
                "beta": [self.beta],
                "mortality": [self.mortality],
                "visibility": [self.visibility],
                "housemate_infection_probability": [h_inf_prob],
            },
            base_params=['grid_size',
                         'N',
                         'customers_in_household',

                         'infected_cashiers_at_start',
                         'percent_of_infected_customers_at_start',
                         ],

            iterations=self.simulations_to_avg,
            max_steps=self.max_sim_days,
            remove_single_results=True
        )

        # As it ran simulations for only one triplet of (visibility, beta, mortality)
        # then result is the one file in avg_directory. To be specific it is
        # the latest file in directory, so read from it.

        latest_file = latest_file_in_dir(directory)
        df = pd.read_csv(latest_file)

        shift, error = self._find_best_x_shift_to_match_plots(
            y1_reference=df['Dead people'],
            y2=RealData.death_tolls().loc[voivodeship],
            y2_start_idx=self.starting_days[voivodeship],
            y2_end_idx=self.ending_days[voivodeship])

        num_of_fitted_days = self.ending_days[voivodeship] - self.starting_days[voivodeship] + 1
        fit_error_per_day = error / num_of_fitted_days

        return shift, fit_error_per_day, latest_file


class BetaOptimizer(Optimizer):
    def __init__(
            self,
            voivodeships: list[str],
            ignored_voivodeships: list[str],
            starting_days_by: StartingDayBy,
            percent_of_touched_counties: int,
            last_date_str_YYYY_mm_dd: str,

            beta: float,
            mortality: float,
            visibility: float,
            housemate_infection_probability: Union[HInfProb, float],

            max_sim_days: int,
            simulations_to_avg: int,
            num_of_shots: int,
    ):
        super().__init__(
            voivodeships=voivodeships,
            ignored_voivodeships=ignored_voivodeships,
            starting_days_by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date_str_YYYY_mm_dd=last_date_str_YYYY_mm_dd,
            beta=beta,
            mortality=mortality,
            visibility=visibility,
            housemate_infection_probability=housemate_infection_probability,
            max_sim_days=max_sim_days,
            simulations_to_avg=simulations_to_avg,
            num_of_shots=num_of_shots,
        )

    def _func_to_optimize(self, vector: np.ndarray, extra_args: dict):
        self.beta = vector[0]
        voivodeship = extra_args['voivodeship']

        shift, fit_error_per_day, avg_fname = self._find_shift_and_fit_error_per_day(voivodeship)

        df = self._create_one_row_df_result_details(
            voivodeship=voivodeship,
            fit_error_per_day=fit_error_per_day,
            shift=shift,
            avg_fname=avg_fname,
            optimize_param=OptimizeParam.BETA
        )

        OptimizeResultsWriterReader.to_csv(df)

        print(f'beta={self.beta * 100:.2f}%, mortality={self.mortality * 100:.2f}%, '
              f'fit_error_per_day={fit_error_per_day:.2f}')

        return fit_error_per_day

    def optimize(self):
        if 'all' in self.voivodeships:
            self.voivodeships = RealData.voivodeships()

        for voivodeship in (set(self.voivodeships) - set(self.ignored_voivodeships)):
            """"
            Init guess (beta in percent not float to be same order as
            mortality to speed up convergence, beta is transformed back later).

            Constraints 'cons' as inequalities 'expression' > 0, example:
            'lambda vec: vec[0] - 1' --> 'x0[0] - 1 > 0' --> 'beta > 1%'.
            Used constraints: (1% < beta < 6%); (1% < mortality < 4%) """
            x0 = np.array([self.beta_init])
            cons = OptimizeParam.BETA_CONS.value
            extra_args = {'voivodeship': voivodeship}

            # Assuming beta_init = 2% -> jumps to 2.25% or 1.75% (rhobeg)
            fmin_cobyla(
                func=self._func_to_optimize,
                x0=x0,
                cons=cons,
                args=(extra_args,),
                consargs=(),
                rhobeg=(1 / 100) * 0.25,  # 0.25% change in variable
                rhoend=(1 / 100) * 0.01,  # 0.01% change
                maxfun=self.num_of_shots,
                catol=0
            )


class MortalityOptimizer(Optimizer):
    def __init__(
            self,
            voivodeships: list[str],
            ignored_voivodeships: list[str],
            starting_days_by: StartingDayBy,
            percent_of_touched_counties: int,
            last_date_str_YYYY_mm_dd: str,

            beta: float,
            mortality: float,
            visibility: float,
            housemate_infection_probability: Union[HInfProb, float],

            max_sim_days: int,
            simulations_to_avg: int,
            num_of_shots: int,
    ):
        super().__init__(
            voivodeships=voivodeships,
            ignored_voivodeships=ignored_voivodeships,
            starting_days_by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date_str_YYYY_mm_dd=last_date_str_YYYY_mm_dd,
            beta=beta,
            mortality=mortality,
            visibility=visibility,
            housemate_infection_probability=housemate_infection_probability,
            max_sim_days=max_sim_days,
            simulations_to_avg=simulations_to_avg,
            num_of_shots=num_of_shots,
        )

    def _func_to_optimize(self, vector: np.ndarray, extra_args: dict):
        self.mortality = vector[0]
        voivodeship = extra_args['voivodeship']

        shift, fit_error_per_day, avg_fname = self._find_shift_and_fit_error_per_day(voivodeship)

        df = self._create_one_row_df_result_details(
            voivodeship=voivodeship,
            fit_error_per_day=fit_error_per_day,
            shift=shift,
            avg_fname=avg_fname,
            optimize_param=OptimizeParam.MORTALITY
        )

        OptimizeResultsWriterReader.to_csv(df)

        print(f'beta={self.beta * 100:.2f}%, mortality={self.mortality * 100:.2f}%, '
              f'fit_error_per_day={fit_error_per_day:.2f}')

        return fit_error_per_day

    def optimize(self):
        if 'all' in self.voivodeships:
            self.voivodeships = RealData.voivodeships()

        for voivodeship in (set(self.voivodeships) - set(self.ignored_voivodeships)):
            """"
            Init guess (beta in percent not float to be same order as
            mortality to speed up convergence, beta is transformed back later).

            Constraints 'cons' as inequalities 'expression' > 0, example:
            'lambda vec: vec[0] - 1' --> 'x0[0] - 1 > 0' --> 'beta > 1%'.
            Used constraints: (1% < beta < 6%); (1% < mortality < 4%) """
            x0 = np.array([self.mortality_init])
            cons = OptimizeParam.MORTALITY_CONS.value
            extra_args = {'voivodeship': voivodeship}

            # Assuming beta_init = 2% -> jumps to 2.25% or 1.75% (rhobeg)
            fmin_cobyla(
                func=self._func_to_optimize,
                x0=x0,
                cons=cons,
                args=(extra_args,),
                consargs=(),
                rhobeg=(1 / 100) * 0.25,  # 0.25% change in variable
                rhoend=(1 / 100) * 0.01,  # 0.01% change
                maxfun=self.num_of_shots,
                catol=0
            )


class BetaMortalityOptimizer(Optimizer):
    def __init__(
            self,
            voivodeships: list[str],
            ignored_voivodeships: list[str],
            starting_days_by: StartingDayBy,
            percent_of_touched_counties: int,
            last_date_str_YYYY_mm_dd: str,

            beta: float,
            mortality: float,
            visibility: float,
            housemate_infection_probability: Union[HInfProb, float],

            max_sim_days: int,
            simulations_to_avg: int,
            num_of_shots: int,
    ):
        super().__init__(
            voivodeships=voivodeships,
            ignored_voivodeships=ignored_voivodeships,
            starting_days_by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date_str_YYYY_mm_dd=last_date_str_YYYY_mm_dd,
            beta=beta,
            mortality=mortality,
            visibility=visibility,
            housemate_infection_probability=housemate_infection_probability,
            max_sim_days=max_sim_days,
            simulations_to_avg=simulations_to_avg,
            num_of_shots=num_of_shots,
        )

    def _func_to_optimize(self, vector: np.ndarray, extra_args: dict):
        self.beta = vector[0]
        self.mortality = vector[1]
        voivodeship = extra_args['voivodeship']

        shift, fit_error_per_day, avg_fname = self._find_shift_and_fit_error_per_day(voivodeship)

        df = self._create_one_row_df_result_details(
            voivodeship=voivodeship,
            fit_error_per_day=fit_error_per_day,
            shift=shift,
            avg_fname=avg_fname,
            optimize_param=OptimizeParam.BETA_AND_MORTALITY
        )

        OptimizeResultsWriterReader.to_csv(df)

        print(f'beta={self.beta * 100:.2f}%, mortality={self.mortality * 100:.2f}%, '
              f'fit_error_per_day={fit_error_per_day:.2f}')

        return fit_error_per_day

    def optimize(self):
        if 'all' in self.voivodeships:
            self.voivodeships = RealData.voivodeships()

        for voivodeship in (set(self.voivodeships) - set(self.ignored_voivodeships)):
            """"
            Init guess (beta in percent not float to be same order as
            mortality to speed up convergence, beta is transformed back later).

            Constraints 'cons' as inequalities 'expression' > 0, example:
            'lambda vec: vec[0] - 1' --> 'x0[0] - 1 > 0' --> 'beta > 1%'.
            Used constraints: (1% < beta < 6%); (1% < mortality < 4%) """
            x0 = np.array([self.beta_init, self.mortality_init])
            cons = OptimizeParam.BETA_AND_MORTALITY_CONS.value
            extra_args = {'voivodeship': voivodeship}

            # Assuming beta_init = 2% -> jumps to 2.25% or 1.75% (rhobeg)
            fmin_cobyla(
                func=self._func_to_optimize,
                x0=x0,
                cons=cons,
                args=(extra_args,),
                consargs=(),
                rhobeg=(1 / 100) * 0.25,  # 0.25% change in variable
                rhoend=(1 / 100) * 0.01,  # 0.01% change
                maxfun=self.num_of_shots,
                catol=0
            )


def main():
    """To make some on fly testing."""

    optimizer_default_params = {
        'voivodeships': RealData.voivodeships(),
        'ignored_voivodeships': ['None'],
        'starting_days_by': StartingDayBy.INFECTIONS,
        'percent_of_touched_counties': 80,
        'last_date_str_YYYY_mm_dd': '2020-07-01',

        'beta': 2.0 / 100,
        'mortality': 2.0 / 100,
        'visibility': 65 / 100,
        'housemate_infection_probability': HInfProb.BY_BETA,

        'max_sim_days': 150,
        'simulations_to_avg': 12,
        'num_of_shots': 10,
    }

    beta_mortality_optimizer = BetaMortalityOptimizer(**optimizer_default_params)
    beta_mortality_optimizer.optimize()

    # beta_optimizer = BetaOptimizer(**optimizer_default_params)
    # beta_optimizer.optimize()

    # mortality_optimizer = MortalityOptimizer(**optimizer_default_params)
    # mortality_optimizer.optimize()


if __name__ == '__main__':
    main()
