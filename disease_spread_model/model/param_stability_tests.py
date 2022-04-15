import copy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mplt
from matplotlib import pyplot as plt
from labellines import labelLines

from disease_spread_model.config import Directories
from disease_spread_model.names import TRANSLATE
from model_runs import RunModel, HInfProb


class Plotter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def plot(self) -> None:
        pass


class StabilityPlotter(Plotter):
    """PLots avg result from csvs in `Logger` with the same `run_purpose`."""

    params_changing_pop_size = {'grid_size', 'N', 'customers_in_household'}

    def __init__(self, run_purpose: str):
        super().__init__()
        self.run_purpose = run_purpose
        self.important_fixed_params = self._get_important_fixed_params()
        self.variable = self._get_variable_param()

        self.plot_values_in_percent = \
            self.variable in self.params_changing_pop_size

    def _get_sub_df_from_csv(self) -> pd.DataFrame:
        """Returns `DataFrame` with given `run_purpose`."""
        df = pd.read_csv(Directories.LOGGING_MODEL_RUNS_FNAME)
        return df[df['run_purpose'] == self.run_purpose]

    def _get_important_fixed_params(self) -> dict:
        """Return `{param_name: value}` for most important params."""
        important_params = {
            'beta', 'mortality', 'visibility', 'N', 'grid_size',
            'infected_cashiers_at_start',
            'percent_of_infected_customers_at_start'
        }

        sub_df = self._get_sub_df_from_csv()
        first_row = sub_df.iloc[0]
        all_params = first_row.to_dict()

        return {k: v for k, v in all_params.items() if k in important_params}

    def _get_variable_param(self) -> str:
        """Return `variable_param` based on `run_purpose` and remove
        it from `self.important_fixed_params`."""

        variable_param = None
        for param in self.important_fixed_params:
            if param in self.run_purpose:
                variable_param = param

        if variable_param:
            self.important_fixed_params.pop(variable_param)
            return variable_param
        else:
            raise ValueError("Can't find `variable_param` in `run_purpose`!")

    @staticmethod
    def _get_pop_size(ser: pd.Series) -> int:
        """Return population size (num of all customers on the grid.)"""

        N = ser['N']
        grid_size = eval(ser['grid_size'])
        customers_in_household = ser['customers_in_household']

        return N * grid_size[0] * grid_size[1] * customers_in_household

    def _get_fdirs(self) -> list[str]:
        """Returns fnames (full path) with given `run_purpose`."""
        return list(self._get_sub_df_from_csv()['avg fname'])

    @staticmethod
    def _create_empty_fig_and_ax() -> (mplt.figure.Figure, mplt.axes.Axes):
        sns.set_style("ticks")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        sns.despine(fig, ax)
        return fig, ax

    @staticmethod
    def _format_values_of_params(params: dict) -> dict[str, str]:
        """
        Return new dict with nicely formatted (ready to put on the plot)
        values of `params`.

        `Params` names passed to this function must be
        in long english form.
        """

        in_params = copy.deepcopy(params)

        # Format those params, other params return unchanged.
        percent_0_digits = {'visibility'}.intersection(in_params)
        percent_2_digits = {'beta', 'mortality'}.intersection(in_params)
        float_0_digits = {
            'N', 'infected_cashiers_at_start'}.intersection(in_params)
        float_3_digits = {
            'percent_of_infected_customers_at_start'}.intersection(in_params)

        pct_0 = {k: f"{in_params[k] * 100:.0f}%" for k in percent_0_digits}
        pct_2 = {k: f"{in_params[k] * 100:.2f}%" for k in percent_2_digits}
        float_0 = {k: f"{in_params[k]:.0f}" for k in float_0_digits}
        float_3 = {k: f"{in_params[k]:.3f}" for k in float_3_digits}

        return in_params | pct_0 | pct_2 | float_0 | float_3

    def _label_axis_and_titles(self,
                               fig: mplt.figure.Figure,
                               ax: mplt.axes.Axes) -> None:
        """Automatically label axis and `ax` and `fig` title."""

        fig.suptitle(
            "Czułość sumy zgonów na " + TRANSLATE.to_polish(self.variable))
        ax.set_xlabel('t, dni')
        ax.set_ylabel(
            'Suma zgonów (% populacji)' if self.plot_values_in_percent
            else 'Suma zgonów')

        fixed_params_formatted = self._format_values_of_params(
            self.important_fixed_params)
        fixed_params_in_polish = TRANSLATE.to_polish(fixed_params_formatted)

        ax_title = ''
        for i, (k, v) in enumerate(fixed_params_in_polish.items()):
            ax_title += f'{k}={v}'
            ax_title += '\n' if (i + 1) % 4 == 0 else '   '
        ax.set_title(ax_title)

    def plot(self):
        """Plot death toll. One line for each avg_df."""
        fig, ax = self._create_empty_fig_and_ax()

        for _, df_row in self._get_sub_df_from_csv().iterrows():
            avg_df = pd.read_csv(df_row['avg fname'])
            death_toll = avg_df['Dead people']
            param_value = df_row[self.variable]

            if self.plot_values_in_percent:
                death_toll /= self._get_pop_size(df_row)
                death_toll *= 100

            label = self._format_values_of_params(
                {self.variable: param_value})[self.variable]
            ax.plot(death_toll, label=label)

        ax.legend(title=TRANSLATE.to_polish(self.variable))
        labelLines(ax.get_lines(), zorder=2)
        self._label_axis_and_titles(fig, ax)

        plt.show()


class Tester(ABC):
    def __init__(self):
        self.model_fixed_params = {
            'grid_size': (30, 30),
            'N': 800,
            'customers_in_household': 3,
            'beta': 2.0 / 100,
            'mortality': 2.0 / 100,
            'visibility': 65 / 100,

            'infected_cashiers_at_start': 30,
            'percent_of_infected_customers_at_start': 0,
            'housemate_infection_probability': HInfProb.BY_BETA,

            'extra_shopping_boolean': True,
            'max_steps': 150,
        }

        self.model_sweep_params = {
            'beta': np.array([2 / 100]),
            'mortality': np.array([2 / 100]),
            'visibility': np.array([65 / 100]),
        }

        self.tester_params = {
            'make_log': True,
            'iterations': 12,
            'run_purpose': 'not specified',
        }

    @abstractmethod
    def _set_run_purpose(self) -> None:
        pass

    @property
    def run_purpose(self) -> str:
        return self.tester_params['run_purpose']

    @abstractmethod
    def run(self) -> None:
        pass


class StabilityTester(Tester, ABC):
    def __init__(self, param_range: np.ndarray):
        super().__init__()
        self._apply_param_range(param_range)

    @abstractmethod
    def _apply_param_range(self, param_range: np.ndarray) -> None:
        pass

    def run(self) -> None:
        """Make simulations and plot Results"""

        self._set_run_purpose()
        RunModel.run_simulation_to_test_sth(
            **self.model_fixed_params,
            sweep_params=self.model_sweep_params,
            **self.tester_params,
        )
        StabilityPlotter(self.run_purpose).plot()


class BetaStabilityTester(StabilityTester):
    def __init__(self, param_range: np.ndarray):
        super().__init__(param_range)

    def _set_run_purpose(self) -> None:
        self.tester_params['run_purpose'] = 'test beta stability'

    def _apply_param_range(self, param_range: np.ndarray) -> None:
        self.model_sweep_params['beta'] = param_range


class MortalityStabilityTester(StabilityTester):
    def __init__(self, param_range: np.ndarray):
        super().__init__(param_range)

    def _set_run_purpose(self) -> None:
        self.tester_params['run_purpose'] = 'test mortality stability'

    def _apply_param_range(self, param_range: np.ndarray) -> None:
        self.model_sweep_params['mortality'] = param_range


class NStabilityTester(StabilityTester):
    def __init__(self, param_range: np.ndarray):
        super().__init__(param_range)

    def _set_run_purpose(self) -> None:
        self.tester_params['run_purpose'] = 'test N stability'

    def _apply_param_range(self, param_range: np.ndarray) -> None:
        self.model_sweep_params['N'] = param_range


class GridSizeStabilityTester(StabilityTester):
    def __init__(self,
                 param_range: np.ndarray,
                 infected_cashiers_at_start: np.ndarray):
        super().__init__(param_range)
        self.nums_of_cashiers = infected_cashiers_at_start
        self.grid_sizes = param_range

    def _set_run_purpose(self) -> None:
        self.tester_params['run_purpose'] = 'test grid_size stability'

    def _apply_param_range(self, param_range: np.ndarray) -> None:
        self.model_sweep_params['grid_size'] = param_range

    def run(self) -> None:
        """Make simulations and plot Results"""

        self._set_run_purpose()

        iterable = zip(self.grid_sizes, self.nums_of_cashiers)
        for grid_size, num_of_cashiers in iterable:
            self.model_sweep_params['grid_size'] = [tuple(grid_size)]
            self.model_sweep_params['infected_cashiers_at_start'] = [
                num_of_cashiers]

            RunModel.run_simulation_to_test_sth(
                **self.model_fixed_params,
                sweep_params=self.model_sweep_params,
                **self.tester_params,
            )

        StabilityPlotter(self.run_purpose).plot()


def run_stability_testers() -> None:
    param_range = np.linspace(2., 3.6, 1) / 100
    # param_range = np.array([100, 200, 500, 800, 1000])
    param_range = np.array([[i, i] for i in range(3, 40, 3)])
    init_cashiers = (np.arange(3, 40, 3) / 3).astype(int)
    # print(f"{param_range = }")

    # BetaStabilityTester(param_range).run()
    # MortalityStabilityTester(param_range).run()
    # NStabilityTester(param_range).run()
    GridSizeStabilityTester(param_range, init_cashiers).run()


def run_stability_plotter() -> None:
    # stability_plotter = StabilityPlotter('test N stability')
    StabilityPlotter('test beta stability').plot()


def main() -> None:
    run_stability_testers()
    # run_stability_plotter()


if __name__ == '__main__':
    main()
