"""Making all kind of plots to visualize real, simulated or both data."""
import datetime
import warnings

from abc import ABC, abstractmethod
from typing import Optional

import matplotlib
import matplotlib.lines as mLines
import matplotlib.pyplot as plt
import matplotlib.patches as mPatches
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mColors
import seaborn as sns

from adjustText import adjust_text
from labellines import labelLines

from scipy.signal import savgol_filter, find_peaks

from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.real_data import RealData
from disease_spread_model.model.model_adjustment import \
    OptimizeResultsWriterReader

from disease_spread_model.model.my_math_utils import *
from disease_spread_model.config import Directories
from disease_spread_model.config import RealDataOptions
from disease_spread_model.config import StartingDayBy


# from disease_spread_model.dirs_to_plot import FolderParams

class Visualisation(ABC):
    BASE_SAVE_DIR = f'{Directories.ABM_DIR}/RESULTS/plots/'

    def __init__(self,
                 show: Optional[bool] = True,
                 save: Optional[bool] = True,
                 ):

        sns.set_style("ticks")
        sns.set_context("talk")  # paper, notebook, talk, poster
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        sns.despine(fig=self.fig, ax=self.ax)

        self.show = show
        self.save = save

        self.save_folder_name = self._set_plot_folder_name()
        self.fname = None
        self.fig_title = None
        self.fig_activated = True

    def _name_of_file_plot_from_title(self) -> str:
        """Creates plot filename from its title."""

        plot_name = self.fig_title.replace('\n', ' ')
        plot_name = plot_name.replace('    ', ' ')
        if plot_name[-1:] == ' ':
            plot_name = plot_name[:-1]

        return plot_name

    def _activate_fig(self) -> None:
        if not self.fig_activated:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111)
            sns.despine(fig=self.fig, ax=self.ax)

    def _set_fig_title(self, title: str) -> None:
        self.fig.suptitle(title)
        self.fig_title = title

    def _show_fig(self) -> None:
        if self.show:
            plt.figure(self.fig)
            plt.show()
            self.fig_activated = False

    def _save_fig(self, file_extension: Optional[str] = 'pdf',
                  **kwargs) -> None:
        """Save plot."""
        if self.save:
            self._set_plot_folder_name()
            self._set_fname(**kwargs)

            plt.figure(self.fig)

            save_dir = self.BASE_SAVE_DIR + self.save_folder_name + '/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + self.fname + '.' + file_extension.lower())

    @abstractmethod
    def _label_plot_components(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _set_plot_folder_name(self) -> str:
        pass

    @abstractmethod
    def _make_basic_plot(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _make_plot_nice_looking(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _make_annotations(self, **kwargs) -> None:
        pass

    @abstractmethod
    def _set_fname(self, **kwargs) -> None:
        pass

    def plot(self) -> None:
        self._make_basic_plot()
        self._make_plot_nice_looking()
        self._make_annotations()

        self._show_fig()
        self._save_fig()


class VisRealDT(Visualisation):
    """
    Plots (normalized) death toll since first day when data were
    collected up to `last_day_do_plot`.

    Death tolls for all voivodeships can be normalized to one
    to show when pandemic started in each voivodeship. Line color
    gradient corresponds to urbanization (red = biggest value)
    """

    def __init__(
            self,
            show: Optional[bool] = True,
            save: Optional[bool] = True,

            last_day_to_plot: Optional[int] = None,
            normalized: Optional[bool] = False,
            voivodeships: Optional[tuple[str, ...]] = ('all',),
    ):
        super().__init__(show, save)
        self.last_day_to_plot = self._get_last_day_to_plot_proper_value(
            last_day_to_plot)
        self.normalized = normalized
        self.voivodeships = voivodeships

    @staticmethod
    def _get_last_day_to_plot_proper_value(
            last_day_to_plot: Optional[int]) -> int:
        if last_day_to_plot:
            return min(last_day_to_plot, len(RealData.death_tolls().columns))
        else:
            return len(RealData.death_tolls().columns)

    def _update_axis_limits(self) -> None:
        """Set nice looking axis limits."""

        if self.normalized:
            self.ax.set_xlim(0, self.last_day_to_plot + 5)
            self.ax.set_ylim(0, 1.1)

    def _get_num_of_voivodeships(self) -> int:
        if 'all' in self.voivodeships:
            return len(RealData.voivodeships())
        else:
            return len(self.voivodeships)

    def _get_colors(self) -> list:
        cmap = plt.get_cmap('rainbow_r')
        num_of_voivodeships = self._get_num_of_voivodeships()

        return [cmap(i) for i in np.linspace(0, 1, num_of_voivodeships)]

    def _set_plot_folder_name(self) -> None:
        self.save_folder_name = 'Death toll dynamic real'

    def _set_fname(self) -> None:
        self.fname = (f'Real DT{" (normalized)" if self.normalized else ""} '
                      f'{self.last_day_to_plot} days')

    def _label_plot_components(self) -> None:
        self.fig.suptitle(self.fig_title)
        self.ax.legend(title="Kolory linii oddają wsp. urbanizacji\n"
                             "(czerwony - najniższa wartość).",
                       fontsize=14, ncol=2)
        self.ax.legend()

        if self.normalized:
            self.ax.set_ylabel("Suma zgonów / max(Suma zgonów)")
        else:
            self.ax.set_ylabel("Suma zgonów")

        self.ax.set_xlabel(
            "t, dni od pierwszego potwierdzonego przypadku w Polsce.")

    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        self._update_axis_limits()
        plt.tight_layout()

    def _make_annotations(self) -> None:

        # annotate day 0 as 2020-03-04
        self.ax.set_xticks(self.ax.get_xticks().tolist()[1:-1])
        self.ax.set_xticklabels(
            [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
             self.ax.get_xticks().tolist()])

    def _make_basic_plot(self) -> None:

        # get sorted urbanization, it will be criteria by which
        # line colors are matched
        urbanization = RealData.get_real_general_data()['urbanization']
        urbanization = urbanization.sort_values()

        real_death_toll = RealData.death_tolls()

        # keep passed voivodeships, but in sorted order
        if 'all' in self.voivodeships:
            voivodeships = urbanization.keys()
        else:
            voivodeships = [
                v for v in urbanization.keys() if v in self.voivodeships]

        for voivodeship, color in zip(voivodeships, self._get_colors()):
            x = list(range(self.last_day_to_plot))
            y = real_death_toll.loc[voivodeship].iloc[:self.last_day_to_plot]

            # normalize y data if possible and needed
            if self.normalized and np.max(y[:self.last_day_to_plot]) > 0:
                y /= np.max(y[: self.last_day_to_plot])

            self.ax.plot(x[: self.last_day_to_plot],
                         y[: self.last_day_to_plot],
                         label=voivodeship, color=color, lw=2)


class VisSummedRealDT(Visualisation):

    def __init__(
            self,
            show: Optional[bool] = True,
            save: Optional[bool] = True,

            last_day_to_plot: Optional[int] = None,
    ):
        super().__init__(show, save)
        self.last_day_to_plot = self._get_last_day_to_plot_proper_value(
            last_day_to_plot)

    @staticmethod
    def _get_last_day_to_plot_proper_value(
            last_day_to_plot: Optional[int]) -> int:
        if last_day_to_plot:
            return min(last_day_to_plot, len(RealData.death_tolls().columns))
        else:
            return len(RealData.death_tolls().columns)

    def _set_plot_folder_name(self) -> None:
        self.save_folder_name = 'Death toll summed dynamic real'

    def _set_fname(self) -> None:
        self.fname = 'Real summed DT'

    def _label_plot_components(self) -> None:
        self.fig.suptitle(self.fig_title)

        self.ax.set_xlabel("t, dni od pierwszego potwierdzonego przypadku.")
        self.ax.set_ylabel("Suma zgonów w Polsce")

    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        plt.tight_layout()

    def _make_annotations(self) -> None:

        # annotate day 0 as 2020-03-04
        self.ax.set_xticks(self.ax.get_xticks().tolist()[1:-1])
        self.ax.set_xticklabels(
            [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
             self.ax.get_xticks().tolist()])

    def _make_basic_plot(self) -> None:
        death_toll = RealData.death_tolls()
        summed_death_toll = death_toll.sum(axis=0)

        x = list(range(self.last_day_to_plot))
        y = summed_death_toll

        self.ax.plot(x[: self.last_day_to_plot],
                     y[: self.last_day_to_plot],
                     lw=2)


class VisSummedRealI(Visualisation):

    def __init__(
            self,
            show: Optional[bool] = True,
            save: Optional[bool] = True,

            last_day_to_plot: Optional[int] = None,
    ):
        super().__init__(show, save)
        self.last_day_to_plot = self._get_last_day_to_plot_proper_value(
            last_day_to_plot)

    @staticmethod
    def _get_last_day_to_plot_proper_value(
            last_day_to_plot: Optional[int]) -> int:
        if last_day_to_plot:
            return min(last_day_to_plot, len(RealData.death_tolls().columns))
        else:
            return len(RealData.death_tolls().columns)

    def _set_plot_folder_name(self) -> None:
        self.save_folder_name = 'Infections summed dynamic real'

    def _set_fname(self) -> None:
        self.fname = 'Real summed infections'

    def _label_plot_components(self) -> None:
        self.fig.suptitle(self.fig_title)

        self.ax.set_xlabel("t, dni od pierwszego potwierdzonego przypadku.")
        self.ax.set_ylabel("Ilość nowych przypadków na dzień w Polsce")

    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        plt.tight_layout()

    def _make_annotations(self) -> None:

        # annotate day 0 as 2020-03-04
        self.ax.set_xticks(self.ax.get_xticks().tolist()[1:-1])
        self.ax.set_xticklabels(
            [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
             self.ax.get_xticks().tolist()])

    def _make_basic_plot(self) -> None:
        infected_tolls = RealData.get_real_infected_toll()
        summed_infected_toll = infected_tolls.sum(axis=0)

        x = list(range(self.last_day_to_plot))
        # Infected not infected toll
        y = [summed_infected_toll[day] if day == 0
             else summed_infected_toll[day] - summed_infected_toll[day - 1]
             for day in x]

        self.ax.plot(x[: self.last_day_to_plot],
                     y[: self.last_day_to_plot],
                     lw=2)


class VisSummedRealD(Visualisation):

    def __init__(
            self,
            show: Optional[bool] = True,
            save: Optional[bool] = True,

            last_day_to_plot: Optional[int] = None,
    ):
        super().__init__(show, save)
        self.last_day_to_plot = self._get_last_day_to_plot_proper_value(
            last_day_to_plot)

    @staticmethod
    def _get_last_day_to_plot_proper_value(
            last_day_to_plot: Optional[int]) -> int:
        if last_day_to_plot:
            return min(last_day_to_plot, len(RealData.death_tolls().columns))
        else:
            return len(RealData.death_tolls().columns)

    def _set_plot_folder_name(self) -> None:
        self.save_folder_name = 'Deaths summed dynamic real'

    def _set_fname(self) -> None:
        self.fname = 'Real summed deaths'

    def _label_plot_components(self) -> None:
        self.fig.suptitle(self.fig_title)

        self.ax.set_xlabel("t, dni od pierwszego potwierdzonego przypadku.")
        self.ax.set_ylabel("Ilość nowych zgonów na dzień w Polsce")

    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        plt.tight_layout()

    def _make_annotations(self) -> None:

        # annotate day 0 as 2020-03-04
        self.ax.set_xticks(self.ax.get_xticks().tolist()[1:-1])
        self.ax.set_xticklabels(
            [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
             self.ax.get_xticks().tolist()])

    def _make_basic_plot(self) -> None:
        death_tolls = RealData.death_tolls()
        summed_death_toll = death_tolls.sum(axis=0)

        x = list(range(self.last_day_to_plot))
        # Infected not infected toll
        y = [summed_death_toll[day] if day == 0
             else summed_death_toll[day] - summed_death_toll[day - 1]
             for day in x]

        self.ax.plot(x[: self.last_day_to_plot],
                     y[: self.last_day_to_plot],
                     lw=2)


class VisRealDTShiftedByHand(Visualisation):
    """
   Makes many death toll plots. On each plot there is death toll for group of similar
   voivodeships. Plots are shifted along X axis in such a way, that pandemic begins
   in `starting_day`.

   Similarity was defined by hand, by looking at death tolls of all voivodeships
   shifted such plots started with chosen value of death toll and looking for what
   initial value of death toll plot smoothly increases.

   If directory_to_data is given than shifted simulated data are also plotted.
   """

    def __init__(self,
                 starting_day: Optional[int] = 10,
                 num_of_days_to_plot: Optional[int] = 100,
                 directory_to_data: Optional[str] = None,
                 ):
        super().__init__()
        self.starting_day = starting_day
        self.num_of_days_to_plot = num_of_days_to_plot
        self.directory_to_data = directory_to_data

    def _set_plot_folder_name(self) -> None:
        self.save_folder_name = 'Death toll dynamic hand shifted'

    def _set_fname(self, common_deaths: int) -> None:
        self.fname = f"Real shifted DT to {common_deaths} deaths, common day = {self.starting_day}"

    def _label_plot_components(self, **kwargs) -> None:
        self.fig.suptitle(self.fig_title)

        common_deaths = kwargs['common_deaths']

        self.ax.set_title(
            f"Suma zgonów w województwach, w których przebieg pandemii, od "
            f"{f'{common_deaths} przypadków' if common_deaths != 1 else 'pierwszego przypadku'} "
            f"był podobny.\n")

        self.ax.set_xlabel(
            f"t, nr. dnia począwszy od dnia w którym "
            f"{'zmarła jedna osoba' if common_deaths == 1 else f'zmarło {common_deaths} osób'} "
            f"w danym województwie")
        self.ax.set_ylabel("Suma zgonów")

    def _make_plot_nice_looking(self) -> None:
        plt.tight_layout()

    def _make_annotations(self) -> None:
        pass

    def _add_simulated_data_to_plot(self, common_deaths: int) -> None:
        """PLot shifted simulated data if dir is given."""
        if not self.directory_to_data:
            return

        def get_colors(_fnames: list) -> list:
            num_of_lines = len(fnames)
            cmap = plt.get_cmap('viridis')
            return [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

        def get_beta_mortality_visibility(_fname: str) -> (
                float, float, float):
            _beta = float(variable_params_from_fname(fname=fname)['beta'])
            _mortality = float(
                variable_params_from_fname(fname=fname)['mortality'])
            _visibility = float(
                variable_params_from_fname(fname=fname)['visibility'])
            return _beta, _mortality, _visibility

        def get_simulated_death_toll(_fname: str) -> np.ndarray:
            df = pd.read_csv(fname)
            return np.array(df['Dead people'])

        def find_common_day_by_shifting_death_toll(
                _death_toll: np.ndarray) -> int:
            try:
                return np.where(_death_toll >= common_deaths)[0][0]
            except IndexError:
                return 0

        def get_label(_beta: float, _mortality: float,
                      _visibility: float) -> str:
            beta_info = r'$\beta$=' + f'{beta}'
            mortality_info = f'mortality={mortality * 100:.1f}%'
            visibility_info = f'visibility={visibility * 100:.0f}%'

            return '{:<10} {:>15} {:>15}'.format(
                beta_info, mortality_info, visibility_info)

        fnames = all_fnames_from_dir(directory=self.directory_to_data)
        for fname, color in zip(fnames, get_colors(fnames)):
            beta, mortality, visibility = get_beta_mortality_visibility(fname)
            death_toll = get_simulated_death_toll(fname)
            common_day = find_common_day_by_shifting_death_toll(death_toll)
            truncated_death_toll = death_toll[common_day - self.starting_day:]
            label = get_label(beta, mortality, visibility)

            self.ax.plot(np.arange(self.num_of_days_to_plot),
                         truncated_death_toll[:self.num_of_days_to_plot],
                         label=label, c=color, lw=1, ls='dashed')

    def _make_basic_plot(self) -> None:
        """
       Makes many death toll plots. On each plot there is death toll for group of similar
       voivodeships. Plots are shifted along X axis in such a way, that pandemic begins
       in starting_day.

       Similarity was defined by hand, by looking at death tolls of all voivodeships
       shifted such plots started with chosen value of death toll and looking for what
       initial value of death toll plot smoothly increases.

       If directory_to_data is given than shifted simulated data are also plotted.
       """

        # dict[starting_deaths] = list(voivodeship1, voivodeship2, ...)
        death_shifts = self._get_dict_shift_to_voivodeships()

        for common_deaths, voivodeships in death_shifts.items():
            death_toll_shifted_df = \
                RealData.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(
                    starting_day=self.starting_day,
                    minimum_deaths=common_deaths)

            self._make_inner_plot(voivodeships, death_toll_shifted_df)
            self._add_simulated_data_to_plot(common_deaths=common_deaths)

            self._label_plot_components(common_deaths=common_deaths)
            self.ax.legend(prop={'family': 'monospace'}, loc='upper left')
            self._make_plot_nice_looking()

            self._save_fig(common_deaths=common_deaths)
            self._show_fig()
            self.fig_activated = False

    def _get_common_start_date(self, voivodeship: str) -> str:
        """Return `2020-03-04 + shift_found_by_hand + starting_day` as str in format YYYY-MM-DD."""

        def get_hand_shift_num_of_days(_voivodeship: str) -> int:
            """Return `shift_found_by_hand`."""
            death_shifts = self._get_dict_shift_to_voivodeships()
            for deaths, voivodeships in death_shifts.items():
                if _voivodeship in voivodeships:
                    shift_date = RealData.get_date_of_first_n_death(deaths)[
                        voivodeship]
                    return RealData.date_to_day(shift_date)
            raise KeyError

        hand_shift_num_of_days = get_hand_shift_num_of_days(voivodeship)
        return RealData.day_to_date(
            day_number=hand_shift_num_of_days + self.starting_day)

    def _make_inner_plot(self,
                         voivodeships: list[str],
                         death_toll_shifted_df: pd.DataFrame,
                         ) -> None:

        self._activate_fig()

        for voivodeship, color in zip(voivodeships,
                                      self._get_colors(voivodeships)):
            x = death_toll_shifted_df.columns  # x = days of pandemic = [0, 1, ...]
            y = death_toll_shifted_df.loc[voivodeship]

            date = f"day {self.starting_day} = {self._get_common_start_date(voivodeship)}"
            label = '{:<20} {:>22}'.format(voivodeship, date)

            self.ax.plot(x[:self.num_of_days_to_plot],
                         y[:self.num_of_days_to_plot],
                         c=color, lw=3, label=label)

    @staticmethod
    def _get_dict_shift_to_voivodeships() -> dict[int: list[str]]:
        """Return dict[starting_deaths_num] = list(voivodeship1, voivodeship2, ...)."""

        voivodeship_starting_deaths_dict = RealData.get_starting_deaths_by_hand()
        unique_death_shifts = sorted(
            list(set(voivodeship_starting_deaths_dict.values())))

        return {
            unique_death_shift:
                [v for v, shift in voivodeship_starting_deaths_dict.items()
                 if shift == unique_death_shift]
            for unique_death_shift in unique_death_shifts
        }

    @staticmethod
    def _get_colors(voivodeships: list) -> list:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(len(voivodeships))]

    def plot(self) -> None:
        self._make_basic_plot()


class VisBestTunedDTDynamic(Visualisation):
    def __init__(self,
                 last_date_to_plot: Optional[str] = '2020-07-01'):

        super().__init__()
        del self.ax
        self.fig.clf()

        gs = GridSpec(2, 1, height_ratios=[3, 1])
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        sns.despine(fig=self.fig, ax=self.ax1)
        sns.despine(fig=self.fig, ax=self.ax2)

        self.color_real = 'C0'
        self.color_simulated = 'C1'

        self.last_date_to_plot = last_date_to_plot
        self.last_day_to_plot = RealData.date_to_day(last_date_to_plot)

    def _activate_fig(self) -> None:
        if not self.fig_activated:
            self.fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 1, height_ratios=[3, 1])
            self.ax1 = self.fig.add_subplot(gs[0, 0])
            self.ax2 = self.fig.add_subplot(gs[1, 0])

            sns.despine(fig=self.fig, ax=self.ax1)
            sns.despine(fig=self.fig, ax=self.ax2)

    def _set_plot_folder_name(self) -> str:
        folder_name = 'Best tuned death toll dynamic'
        self.save_folder_name = folder_name

        return folder_name

    def _set_fname(self, voivodeship) -> None:
        self.fname = f"{voivodeship.capitalize()} DT dyn best tuned"

    def _label_plot_components(self, voivodeship: str, beta: float,
                               mortality: float, visibility: float,
                               h_inf_prob: float,
                               frac_init_infected_customer: float,
                               ) -> None:

        self.ax1.set_title('Zgony')
        self.ax1.set_xlabel('t, dni')
        self.ax1.set_ylabel('Suma zgonów')

        self.ax2.set_title('Wyzdrowienia')
        self.ax2.set_xlabel('t, dni')
        self.ax2.set_ylabel('Suma wyzdrowień')

        self._set_fig_title(
            f'{voivodeship.capitalize()}, rezultat dopasowania parametrów modelu.\n'
            r"$\beta_{spotkanie}^{klient-kasjer}$"f'={beta * 100:.2f}%, '
            r"$\beta_{całkowita}^{współlokatorzy}$"f'={h_inf_prob * 100:.2f}%, ' '\n'
            f'początkowy udział zarażonych klientów={frac_init_infected_customer * 1000:.2f}‰, ' "\n"
            f'śmiertelność={mortality * 100:.2f}%, '
            f'widoczność={visibility * 100:.0f}%')

    @staticmethod
    def _get_day_sim_values_dict(simulated_toll: np.ndarray,
                                 shift: int) -> dict[int: float]:
        """Example of return `{-1: 0, 0: 1, ..., 90: 800, 91: 832}` {true_day: simulated_toll}."""
        return {
            true_day: simulated_toll[idx]
            for idx, true_day in enumerate(
                np.arange(len(simulated_toll)) + shift
            )
        }

    def _plot_and_annotate_data(self,
                                real_death_toll: np.ndarray,
                                simulated_death_toll: np.ndarray,
                                real_recovered_toll: np.ndarray,
                                simulated_recovered_toll: np.ndarray,
                                shift: int,
                                num_of_first_day: int,
                                num_of_last_day: int,
                                ) -> None:

        # Truncate data up to `last_day_to_plot`
        day_real_death_toll = self._get_day_sim_values_dict(real_death_toll, 0)
        day_real_recovered_toll = self._get_day_sim_values_dict(
            real_recovered_toll, 0)
        day_simulated_death_toll = self._get_day_sim_values_dict(
            simulated_death_toll, shift)
        day_simulated_recovered_toll = self._get_day_sim_values_dict(
            simulated_recovered_toll, shift)

        self.ax1.plot(
            [item[0] for item in day_real_death_toll.items() if
             item[0] <= self.last_day_to_plot],
            [item[1] for item in day_real_death_toll.items() if
             item[0] <= self.last_day_to_plot],
            label='raportowana suma zgonów', c=self.color_real, lw=2, zorder=1)

        self.ax1.plot(
            [item[0] for item in day_simulated_death_toll.items() if
             item[0] <= self.last_day_to_plot],
            [item[1] for item in day_simulated_death_toll.items() if
             item[0] <= self.last_day_to_plot],
            label='symulowana suma zgonów', c=self.color_simulated, lw=5,
            alpha=0.6, zorder=1)

        self.ax2.plot(
            [item[0] for item in day_real_recovered_toll.items() if
             item[0] <= self.last_day_to_plot],
            [item[1] for item in day_real_recovered_toll.items() if
             item[0] <= self.last_day_to_plot],
            label='raportowana suma zgonów', c=self.color_real, lw=2, zorder=1)

        self.ax2.plot(
            [item[0] for item in day_simulated_recovered_toll.items() if
             item[0] <= self.last_day_to_plot],
            [item[1] for item in day_simulated_recovered_toll.items() if
             item[0] <= self.last_day_to_plot],
            label='symulowana suma zgonów', c=self.color_simulated, lw=5,
            alpha=0.6, zorder=1)

        self._make_annotations(num_of_first_day=num_of_first_day,
                               num_of_last_day=num_of_last_day,
                               day_real_death_toll=day_real_death_toll,
                               day_real_recovered_toll=day_real_recovered_toll,
                               day_simulated_recovered_toll=day_simulated_recovered_toll)

        self._make_plot_nice_looking(day_real_death_toll=day_real_death_toll)

    def _make_basic_plot(self,
                         _df_best_run_details: pd.Series,
                         _real_death_toll: np.ndarray,
                         _real_recovered_toll: np.ndarray) -> None:

        sim_avg_df = pd.read_csv(_df_best_run_details['fname'])

        self._plot_and_annotate_data(real_death_toll=_real_death_toll,
                                     real_recovered_toll=_real_recovered_toll,
                                     simulated_death_toll=sim_avg_df[
                                         'Dead people'],
                                     simulated_recovered_toll=sim_avg_df[
                                         'Recovery people'],
                                     shift=int(_df_best_run_details['shift']),
                                     num_of_first_day=int(
                                         _df_best_run_details['first day']),
                                     num_of_last_day=int(
                                         _df_best_run_details['last day']),
                                     )

        self._label_plot_components(
            voivodeship=_df_best_run_details['voivodeship'],
            beta=_df_best_run_details['beta'],
            mortality=_df_best_run_details['mortality'],
            visibility=_df_best_run_details['visibility'],
            h_inf_prob=_df_best_run_details['h inf prob'],
            frac_init_infected_customer=_df_best_run_details[
                'fraction of infected customers at start']
        )

        plt.tight_layout()

    def _make_plot_nice_looking(self,
                                day_real_death_toll: dict[int, float]) -> None:
        """
        Annotate day 0 as 2020-03-04.
        Set reasonable y_ax1is limits
        """

        def improve_ax1_axis():
            # Annotate day 0 as 2020-03-04
            self.ax1.set_xticks(self.ax1.get_xticks().tolist()[1:-1])
            self.ax1.set_xticklabels(
                [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
                 self.ax1.get_xticks().tolist()])

            max_plotted_death_toll_real = day_real_death_toll[
                self.last_day_to_plot]
            if max_plotted_death_toll_real != 0:
                self.ax1.set_ylim([-0.1 * max_plotted_death_toll_real,
                                   1.5 * max_plotted_death_toll_real])

        def improve_ax2_axis():
            def my_number_formatter(*args):
                value = args[0]
                return f"{value * 0.001:.0f}k" if value != 0 else f"{value:.0f}"

            # Format number on y_axis
            self.ax2.yaxis.set_major_formatter(my_number_formatter)

            # Annotate day 0 as 2020-03-04
            self.ax2.set_xticks(self.ax2.get_xticks().tolist()[1:-1])
            self.ax2.set_xticklabels(
                [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
                 self.ax2.get_xticks().tolist()])

        improve_ax1_axis()
        improve_ax2_axis()
        plt.tight_layout()

    def _make_annotations(self,
                          num_of_first_day: int,
                          num_of_last_day: int,
                          day_real_death_toll: dict[int, float],
                          day_real_recovered_toll: dict[int, float],
                          day_simulated_recovered_toll: dict[int, float],
                          ) -> None:

        def annotate_ax1() -> None:
            self.ax1.axvline(num_of_first_day, c='red')
            self.ax1.axvline(num_of_last_day, c='red',
                             label='dni do których dopasowywano model')

            max_death_toll_real = day_real_death_toll[self.last_day_to_plot]
            # Annotate how many days was fitted (draw arrow)
            self.ax1.annotate("",
                              xy=(num_of_first_day, max_death_toll_real),
                              xytext=(num_of_last_day, max_death_toll_real),
                              arrowprops=dict(arrowstyle="<->"))

            # Annotate how many days was fitted (write days over arrow)
            self.ax1.annotate(
                f"{num_of_last_day - num_of_first_day} dni",
                xy=((num_of_last_day + num_of_first_day) / 2,
                    max_death_toll_real * 1.02),
                ha='center', va='bottom', fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.8,),
                          # set `alpha=0.8`
                          ec="none"),
            )

            # Annotate starting fit date (text nearest to first axvline)
            self.ax1.text(
                x=num_of_first_day,
                y=max_death_toll_real / 2,
                s=RealData.day_to_date(num_of_first_day),
                rotation=90, ha='right', va='center', fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            # Annotate ending fit date (text nearest to second axvline)
            self.ax1.text(
                x=num_of_last_day,
                y=max_death_toll_real / 2,
                s=RealData.day_to_date(num_of_last_day),
                rotation=90, ha='left', va='center', fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            self.ax1.legend(loc='upper left', framealpha=1)

        def annotate_ax2() -> None:
            self.ax2.axvline(num_of_last_day, c='red')
            sim_max_recovered_toll_plotted = day_simulated_recovered_toll[
                self.last_day_to_plot]
            y_annotate_offset = .1 * sim_max_recovered_toll_plotted

            # Annotate real_recovered_toll in last day of fit (with arrow and value)
            self.ax2.annotate(
                f"{day_real_recovered_toll[num_of_last_day] / 1000:.1f}k",
                xy=(num_of_last_day, day_real_recovered_toll[num_of_last_day]),
                xytext=(num_of_last_day - 10,
                        day_real_recovered_toll[
                            num_of_last_day] + .5 * y_annotate_offset),
                c=self.color_real, fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none",
                          lw=2))

            # Annotate simulated_recovered_toll in last day of fit (with arrow and value)
            self.ax2.annotate(
                f"{day_simulated_recovered_toll[num_of_last_day] / 1000:.1f}k",
                xy=(num_of_last_day,
                    day_simulated_recovered_toll[num_of_last_day]),
                xytext=(num_of_last_day - 10,
                        day_simulated_recovered_toll[
                            num_of_last_day] + 2 * y_annotate_offset),
                c=self.color_simulated, fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none",
                          lw=2))

            self.ax2.legend(loc='upper left')

        annotate_ax1()
        annotate_ax2()

    def plot(self):
        df_real_death_tolls = RealData.death_tolls()
        df_real_recovered_tolls = RealData.recovered_tolls()
        df_best_tuned_csv_details = OptimizeResultsWriterReader.get_n_best_tuned_results(
            n=1)

        for _, df_best_run_details in df_best_tuned_csv_details.iterrows():
            self._activate_fig()
            voivodeship = df_best_run_details['voivodeship']

            self._make_basic_plot(
                _df_best_run_details=df_best_run_details,
                _real_death_toll=np.array(
                    df_real_death_tolls.loc[voivodeship]),
                _real_recovered_toll=np.array(
                    df_real_recovered_tolls.loc[voivodeship])
            )

            self._save_fig(voivodeship=voivodeship)
            self._show_fig()


class RealVisualisation:
    """
    Class responsible for making visualisation of real data only.

    Methods:
        * show_real_death_toll --> plots  death toll since first day when data were
            collected up to some given day. Very basic function.

        * show_real_death_toll_shifted_by_hand --> makes many death toll plots.
            On each plot there is death toll for group of similar voivodeships.
            Plots are shifted along X axis in such a way, that pandemic begins
            in starting_day.

            Similarity was defined by hand, by looking at death tolls of all voivodeships
            shifted such plots started with chosen value of death toll and looking for what
            initial value of death toll plot smoothly increases.

        * plot_pandemic_starting_days_by_touched_counties --> make one figure where:
            X axis - voivodeship
            Y1 axis - day number since 04-03-2020, which is considered as first day of
                pandemic in given voivodeship, based od percentage of counties in which
                died at least one pearson
            Y2 axis - death toll in day given by Y1

            Series:
                + day number since 04-03-2020, which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    died at least one pearson.
                + day number since 04-03-2020, which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    died at least one pearson.
                + day number since 04-03-2020, which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    at least one person fell ill.
                + death tolls correlated to both dates.


    """

    @classmethod
    def __show_and_save(cls, fig, plot_type, plot_name, save, show,
                        file_format='pdf'):
        """Function that shows and saves figures.

        Parameters:
            :param fig: figure to be shown/saved,
            :type fig: matplotlib figure,
            :param plot_type: general description of a plot type, each unique type will have its own folder,
            :type plot_type: str,
            :param plot_name: detailed name of plot, it will serve as filename,
            :type plot_name: str,
            :param save: save plot?
            :type save: bool,
            :param show: show plot?
            :type show: bool,
        """

        # set fig as current figure
        plt.figure(fig)
        if save:
            save_dir = f'{Directories.ABM_DIR}/RESULTS/plots/{plot_type}/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + plot_name + '.' + file_format.lower())

        if show:
            plt.show()

        plt.close(fig)

    @classmethod
    def _get_plot_name_from_title(cls, title: str):
        """
        Creates plot filename from its title.

        Parameters
        ----------
        title : str
            plot title as it is seen by plt.show()

        Returns
        -------
        plot_name : str
            plot_name which can be used as plot filename
        """

        plot_name = title
        plot_name = plot_name.replace('\n', ' ')
        plot_name = plot_name.replace('    ', ' ')
        if plot_name[-1:] == ' ':
            plot_name = plot_name[:-1]

        return plot_name

    @staticmethod
    def _annotate_day_0_on_X_axis_as_date(
            ax: matplotlib.axes,
            date0: datetime.datetime) -> None:

        ax.set_xticks(ax.get_xticks().tolist()[1:-1])
        ax.set_xticklabels(
            [f'{x:.0f}' if x != 0 else f'{date0: %Y-%m-%d}' for x in
             ax.get_xticks().tolist()])

    @classmethod
    def show_real_death_toll_shifted_by_hand(cls,
                                             starting_day=10,
                                             day_in_which_colors_are_set=60,
                                             last_day=100,
                                             directory_to_data=None,
                                             shift_simulated=False,
                                             save=False,
                                             show=True):  # sourcery no-metrics
        """
        Makes many death toll plots. On each plot there is death toll for group of similar
        voivodeships. Plots are shifted along X axis in such a way, that pandemic begins
        in starting_day.

        Similarity was defined by hand, by looking at death tolls of all voivodeships
        shifted such plots started with chosen value of death toll and looking for what
        initial value of death toll plot smoothly increases.

        If directory_to_data is given than shifted simulated data are also plotted.
        """

        warnings.warn(
            "Please use `VisRealDTShiftedByHand.plot()` instead of this.",
            DeprecationWarning)

        # make dict: dict[starting_deaths] = list(voivodeship1, voivodeship2, ...) *****************************
        voivodeship_starting_deaths = RealData.get_starting_deaths_by_hand()
        unique_death_shifts = sorted(
            list(set(voivodeship_starting_deaths.values())))
        death_shifts = {
            death_shift: [
                voivodeship
                for voivodeship, val in voivodeship_starting_deaths.items()
                if val == death_shift
            ]
            for death_shift in unique_death_shifts
        }

        # ****************************************************************************************************

        # for each pair (minimum_deaths, [voivodeship1, voivodeship2, ..]
        for minimum_deaths, voivodeships in death_shifts.items():
            shifted_real_death_toll = \
                RealData.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(
                    starting_day=starting_day,
                    minimum_deaths=minimum_deaths)

            true_start_day = RealData.get_date_of_first_n_death(
                n=minimum_deaths)

            # df_indices_order[voivodeship] = num of voivodeship sorted shifted_real_death_toll
            # by death toll in day = 60. Used to determine colors of lines in the plot.
            df_indices_order = sort_df_indices_by_col(
                df=shifted_real_death_toll, column=day_in_which_colors_are_set)
            death_toll_final_order = {}
            i = 0
            for voivodeship in df_indices_order:
                if voivodeship in voivodeships:
                    death_toll_final_order[voivodeship] = i
                    i += 1

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.set_title(
                f"Death toll for ,,similar'' voivodeships, shifted in such a way, that in day {starting_day} "
                f"death toll is not less than {minimum_deaths}.\n"
                f"Mapping: (voivodeship, line color) was performed in day {day_in_which_colors_are_set} "
                f"with respect to death toll in that day.")
            ax.set_xlabel(
                f't, days since first {minimum_deaths} people died in given voivodeship')
            ax.set_ylabel(
                f'Death toll (since first {minimum_deaths} people died in given voivodeship)')

            num_of_lines = len(voivodeships)
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

            for voivodeship, index in death_toll_final_order.items():
                x = shifted_real_death_toll.columns  # x = days of pandemic = [0, 1, ...]
                y = shifted_real_death_toll.loc[voivodeship]

                left = voivodeship
                right = f"day {starting_day} = {true_start_day[voivodeship]}"
                label = '{:<20} {:>22}'.format(left, right)

                ax.plot(x[:last_day], y[:last_day],
                        label=label,
                        color=colors[index],
                        linewidth=3)

            if directory_to_data:
                fnames = all_fnames_from_dir(directory=directory_to_data)

                num_of_lines = len(fnames)
                cmap = plt.get_cmap('vir    idis')
                colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

                for i, fname in enumerate(fnames):
                    beta = float(
                        variable_params_from_fname(fname=fname)['beta'])
                    mortality = float(
                        variable_params_from_fname(fname=fname)['mortality'])
                    visibility = float(
                        variable_params_from_fname(fname=fname)['visibility'])

                    df = pd.read_csv(fname)

                    common_day = 0
                    if shift_simulated:
                        y = np.array(df['Dead people'])
                        try:
                            common_day = np.where(y >= minimum_deaths)[0][0]
                        except IndexError:
                            common_day = 0
                        y = y[common_day - starting_day:]
                        x = list(range(len(y)))
                    else:
                        x = df['Day']
                        y = df['Dead people']

                    beta_info = r'$\beta$=' + f'{beta}'
                    mortality_info = f'mortality={mortality * 100:.1f}%'
                    visibility_info = f'visibility={visibility * 100:.0f}%'
                    day_info = f"day {starting_day}=0"
                    if shift_simulated:
                        day_info = f"day {starting_day}={common_day}"

                    label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                                 mortality_info,
                                                                 visibility_info,
                                                                 day_info)

                    ax.plot(x[:last_day], y[:last_day], label=label,
                            color=colors[i],
                            linewidth=1, linestyle='dashed')

            ax.legend(prop={'family': 'monospace'}, loc='upper left')
            plt.tight_layout()

            if save:
                plot_type = 'Real death toll for similar voivodeships, shifted by hand'
                save_dir = 'results/plots/' + plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = f'minimum_deaths={minimum_deaths},   ' \
                       f'starting_day={starting_day},   ' \
                       f'day_in_which_colors_are_set={day_in_which_colors_are_set}   ' \
                       f'last_day={last_day}'
                plt.savefig(save_dir + name + '.pdf')

            if show:
                plt.show()
            plt.close(fig)

    @classmethod
    def plot_pandemic_starting_days_by_touched_counties(cls,
                                                        percent_of_death_counties: int,
                                                        percent_of_infected_counties: int,
                                                        normalize_by_population: bool,
                                                        save=False,
                                                        show=True):
        """
        Plots first day of pandemic for all voivodeships since data were collected.
        ALso plots death toll in that day.

        """

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # PLot first (main) plot on main ax (starting day) **********************************************************
        # make title and labels
        main_info = ("Day that is consider as starting day of pandemic "
                     "for given voivodeship and death toll in that day.\n"
                     " Day 0 is 04-03-2020.")

        ax.set_title(main_info)
        ax.set_xlabel('Voivodeship')
        ax.set_ylabel(
            'Day number since 04-03-2020 which is considered to be the beginning of a pandemic')

        # get starting day of pandemic by percent of touched counties
        starting_days_deaths = RealData.starting_days(
            by='deaths',
            percent_of_touched_counties=percent_of_death_counties,
            ignore_healthy_counties=False)

        starting_days_infections = RealData.starting_days(
            by='infections',
            percent_of_touched_counties=percent_of_infected_counties,
            ignore_healthy_counties=False)

        # get voivodeships and used them to synchronize plots
        voivodeships_synchro = starting_days_infections.keys()
        color_infections = 'lime'
        color_deaths = 'deepskyblue'

        # plot starting_days_infections
        l1 = ax.plot(voivodeships_synchro,
                     [starting_days_infections[voivodeship] for voivodeship in
                      voivodeships_synchro],
                     color=color_infections, linestyle='-.', marker='o',
                     mec='black',
                     label=(
                         f'Starting day by {percent_of_infected_counties}% of counties with '
                         f'at least one infected case.'))

        # plot starting_days_deaths
        l2 = ax.plot(voivodeships_synchro,
                     [starting_days_deaths[voivodeship] for voivodeship in
                      voivodeships_synchro],
                     color=color_deaths, linestyle='-.', marker='o',
                     mec='black',
                     label=(
                         f'Starting day by {percent_of_death_counties}% of counties with '
                         f'at least one death case.'))

        # set y_lim to 0
        ax.set_ylim([-5, None])

        # rotate label of outer x-axis
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        # **************************************************************************************************

        # Plot second plot on the other y-axis (death toll in starting day) ********************************
        # get starting death toll by starting day by percent of touched counties
        starting_death_toll_deaths = RealData.get_starting_death_toll_for_voivodeships_by_days(
            voivodeships_days=starting_days_deaths)

        starting_death_toll_infections = RealData.get_starting_death_toll_for_voivodeships_by_days(
            voivodeships_days=starting_days_infections)

        # second y axis
        ax2 = ax.twinx()

        lab_death_toll_deaths = 'Death toll in starting day (by deaths).'
        lab_death_toll_infections = 'Death toll in starting day (by infections).'

        # plot death toll on the second y-axis (normalized or not)
        if normalize_by_population:
            y_label2 = '(Death toll / population) ' r'$\cdot 10^5$'
            population = RealData.get_real_general_data()['population']
            # preserve order of voivodeship on X axis

            p3 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_infections[voivodeship] /
                              population[voivodeship] * (10 ** 5)
                              for voivodeship in voivodeships_synchro],
                             color=color_infections,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_infections)

            p4 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_deaths[voivodeship] /
                              population[voivodeship] * (10 ** 5)
                              for voivodeship in voivodeships_synchro],
                             color=color_deaths,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_deaths)
        else:
            y_label2 = 'Death toll (in given day)'

            p3 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_infections[voivodeship] for
                              voivodeship in voivodeships_synchro],
                             color=color_infections,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_infections)

            p4 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_deaths[voivodeship] for
                              voivodeship in voivodeships_synchro],
                             color=color_deaths,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_deaths)

        # set y2 axis label
        ax2.set_ylabel(y_label2)
        # ***********************************************************************************************************

        # add legend (both y-axis have common legend)
        plots = [l1, l2, p3, p4]
        # for some reason scatter plot is a standard object, but line plot is a list containing lines
        plots = [p if type(p) != list else p[0] for p in plots]
        labs = [p.get_label() for p in plots]
        ax.legend(plots, labs)
        plt.tight_layout()

        cls.__show_and_save(fig=fig,
                            plot_type='Starting days for voivodeships based on touched district',
                            plot_name=(
                                f'percent_of_death_counties={percent_of_death_counties},   '
                                f'percent_of_infected_counties={percent_of_infected_counties},   '
                                f'normalize_by_population={normalize_by_population}'),
                            save=save,
                            show=show)

    @classmethod
    def plot_last_day_finding_process(
            cls,
            voivodeships=('all',),
            start_days_by=StartingDayBy.INFECTIONS,
            percent_of_touched_counties=80,
            last_date='2020-07-01',
            death_toll_smooth_out_win_size=21,
            death_toll_smooth_out_polyorder=3,
            derivative_half_win_size=3,
            show=True,
            save=False,
    ):  # sourcery no-metrics
        """
        Plots crucial steps in finding last day of pandemic in voivodeships.

        Plots:
         - death tool
         - death tool smoothed
         - slope of smoothed up death toll
        """

        # get voivodeships
        if 'all' in voivodeships:
            voivodeships = RealData.voivodeships()

        # get start days dict
        start_days = RealData.starting_days(
            by=start_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)

        # get real death toll for all voivodeships (since 03.04.2020)
        death_tolls = RealData.death_tolls()

        # fill gaps in real death toll
        for voivodeship in RealData.voivodeships():
            death_tolls.loc[voivodeship] = complete_missing_data(
                values=death_tolls.loc[voivodeship])

        # smooth out death toll
        death_tolls_smooth = death_tolls.copy()
        for voivodeship in RealData.voivodeships():
            death_tolls_smooth.loc[voivodeship] = savgol_filter(
                x=death_tolls.loc[voivodeship],
                window_length=death_toll_smooth_out_win_size,
                polyorder=death_toll_smooth_out_polyorder)

        # get last day in which death pandemic last day will be looked for
        last_day_to_search = RealData.date_to_day(date=last_date)

        extra_days = 10
        # Make plots
        for voivodeship in voivodeships:
            sns.set_style("ticks")
            sns.set_context("poster")  # paper, notebook, talk, poster
            # sns.set_context("talk")  # paper, notebook, talk, poster
            fig, ax = plt.subplots(figsize=(12, 8))
            ax2 = ax.twinx()

            day0 = int(start_days[voivodeship])

            # death toll
            DT_days = np.arange(-extra_days, last_day_to_search - day0)
            death_toll = death_tolls.loc[voivodeship]
            ax.plot(
                DT_days,
                death_toll[day0 - extra_days: last_day_to_search],
                color='C0', label='suma zgonów', lw=4, alpha=0.7,
            )

            # death toll smoothed up
            death_toll_smooth = death_tolls_smooth.loc[voivodeship]
            ax.plot(
                DT_days,
                death_toll_smooth[day0 - extra_days: last_day_to_search],
                color='C1', label='suma zgonów po wygładzeniu',
            )

            # Set new ticks on x-axis.
            locations = np.arange(last_day_to_search - day0, step=15)
            date0 = RealData.day_to_date(day0)
            labels = [date0 if loc == 0 else loc for loc in locations]
            ax.xaxis.set_ticks(locations, labels)

            # slope of smooth death toll
            slope_smooth = slope_from_linear_fit(
                data=death_toll_smooth,
                half_win_size=derivative_half_win_size)
            slope_smooth = slope_smooth[day0: last_day_to_search]

            # normalize slope to 1
            if max(slope_smooth) > 0:
                slope_smooth /= max(slope_smooth)

            # plot normalized slope
            ax2.plot(
                slope_smooth,
                color='green', lw=4,
                label='nachylenie prostej dopasowanej do'
                      ' fragmentów wygładzonej sumy zgonów')

            # Proxy plot to keep y-axis upper limit to 1 in case if there
            # are not any deaths.
            ax2.plot([1], [1])

            # Maxima of slope
            last_day_candidates = find_peaks(
                slope_smooth, distance=8, height=.5)[0]
            ax2.scatter(last_day_candidates, slope_smooth[last_day_candidates],
                        color='lime', s=140, zorder=100,
                        label='maksima nachylenia sumy zgonów')

            # Last day of pandemic as day of peak closest to day 60
            # if there are no candidates add day with the largest peak
            if len(last_day_candidates) == 0:
                last_day_candidates = np.array([0])

            # choose find last day (nearest to 60) from candidates
            last_day_since_day0 = min(
                last_day_candidates, key=lambda x: abs(x - 60))

            # Add 7 days, because death toll used to change rapidly right
            # after 7 days since max death toll slope occurred.
            last_day_since_day0 += 7

            # Plot and annotate line representing last day of pandemic
            ax.axvline(last_day_since_day0, color='red')
            ax2.text(
                x=last_day_since_day0,
                y=0.15,
                s=f'{datetime.datetime.fromisoformat(date0) + datetime.timedelta(days=int(last_day_since_day0)): %Y-%m-%d}',
                rotation=90, ha='right', va='center', fontsize=30,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            ymin, ymax = ax.get_ylim()
            ax.set_ylim([0 - ymax * 0.1, ymax * 1.2])

            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim([0 - ymax * 0.1, ymax * 1.2])

            # ax.set_xlabel('t, dni')
            # ax.set_ylabel('Suma zgonów')
            # ax2.set_ylabel('Przeskalowane nachylenie sumy zgonów')

            ax2.text(
                x=-extra_days, y=1.15, fontsize=30,
                s=f'{voivodeship.capitalize()}, {last_day_since_day0} dni',
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            # fig.legend(
            #     loc="upper center",
            #     ncol=1,
            #     fontsize=15,
            #     bbox_to_anchor=(0.5, 1),
            #     bbox_transform=ax.transAxes,
            #     fancybox=True,
            #     shadow=True)

            plt.tight_layout()
            cls.__show_and_save(
                fig=fig,
                plot_type=f'Finding last day of pandemic up to {last_date}',
                plot_name=(
                    f'{voivodeship}, by {start_days_by} in {percent_of_touched_counties} '
                    f'percent of counties'),
                save=save,
                show=show,
                file_format='pdf')

    @classmethod
    def plot_pandemic_time(cls,
                           starting_days_by=RealDataOptions.STARTING_DAYS_BY,
                           percent_of_touched_counties=80,
                           death_toll_smooth_out_win_size=21,
                           death_toll_smooth_out_polyorder=3,
                           last_date='2020-07-01',
                           save=False,
                           show=True):
        """
        Plots first and last day of pandemic for all voivodeships
        since data were collected. Start day can be deduced by percent of
        death or infected counties. Last day got from slope of death toll.
        """

        # make fig and ax
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # make ax title and axis labels
        main_info = (
            f"Days considered as the period of the first phase of the pandemic.\n"
            f"First day found based on percentage of counties with at least one "
            f"{'infection' if starting_days_by == StartingDayBy.INFECTIONS else 'death'} cases "
            f"({percent_of_touched_counties}%)."
            f" Day 0 is 04-03-2020.")

        ax.set_title(main_info)
        ax.set_xlabel('Voivodeship')
        ax.set_ylabel('Days of first phase of pandemic')

        # get start days dict
        starting_days = RealData.starting_days(
            by=starting_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)

        ending_days = RealData.ending_days_by_death_toll_slope(
            starting_days_by=RealDataOptions.STARTING_DAYS_BY.INFECTIONS,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date=last_date,
            death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
            death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
        )

        # Get dict {voivodeship: pandemic_duration_in_days}
        pandemic_duration = {
            voivodeship: ending_days[voivodeship] - starting_days[voivodeship]
            for voivodeship in starting_days.keys()
        }

        # Sort dict by values
        pandemic_duration = sort_dict_by_values(pandemic_duration)

        # Get voivodeships and used them to synchronize plots.
        # Plots ordered by pandemic duration in voivodeships
        voivodeships_synchro = pandemic_duration.keys()

        # Plot days when first phase of pandemic was observed
        for voivodeship in voivodeships_synchro:
            x = voivodeship
            y_up = ending_days[voivodeship]
            y_down = starting_days[voivodeship]
            markerline, stemlines, baseline = ax.stem(x,
                                                      y_up,
                                                      bottom=y_down,
                                                      basefmt='C0o')
            markerline.set_markersize(15)
            baseline.set_markersize(15)
            stemlines.set_linewidth(6)

        # rotate x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        # improve y lower limit
        ax.set_ylim([0, None])

        plt.tight_layout()
        cls.__show_and_save(fig=fig,
                            plot_type='Days of pandemic',
                            plot_name=(
                                f'Pandemic time based on {starting_days_by.name}, '
                                f'percentage {percent_of_touched_counties}'),
                            save=save,
                            show=show)

    @classmethod
    def compare_pandemic_time_by_infections_and_deaths(cls,
                                                       percent_of_deaths_counties=20,
                                                       percent_of_infected_counties=80,
                                                       death_toll_smooth_out_win_size=21,
                                                       death_toll_smooth_out_polyorder=3,
                                                       last_date='2020-07-01',
                                                       save=False,
                                                       show=True):
        """
        Plots first day and last day of pandemic for all voivodeships
        since data were collected. One series where start day was deduced
        by percent of death counties, second based on infections.

        :param last_date: last possible date, which can be last day of pandemic
        :type last_date: str
        :param death_toll_smooth_out_polyorder: savgol_filter polyorder
        :type death_toll_smooth_out_polyorder: int
        :param death_toll_smooth_out_win_size: savgol_filter win size
        :type death_toll_smooth_out_win_size: int
        :param percent_of_deaths_counties: percent of counties in which
            someone died
        :type percent_of_deaths_counties: int
        :param percent_of_infected_counties: percent of counties in which
            someone got infected
        :type percent_of_infected_counties: int
        :param save: save?
        :type save: bool
        :param show: show?
        :type show: bool
        """

        # make fig and ax
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        # make ax title and axis labels
        main_info = (
            "Summary of designated positions in time of the first stage of the pandemic.\n"
            "First day found based on percentage of counties with at least one "
            "infection or death case. Day 0 is 04-03-2020")

        ax.set_title(main_info)
        ax.set_xlabel('Voivodeship')
        ax.set_ylabel('Days of first phase of pandemic')

        # get starting day of pandemic by death counties
        starting_days_by_deaths = RealData.starting_days(
            by='deaths',
            percent_of_touched_counties=percent_of_deaths_counties,
            ignore_healthy_counties=False)

        # get starting day of pandemic by infected counties
        starting_days_by_infections = RealData.starting_days(
            by='deaths',
            percent_of_touched_counties=percent_of_infected_counties,
            ignore_healthy_counties=False)

        ending_days_by_deaths = \
            RealData.ending_days_by_death_toll_slope(
                starting_days_by=StartingDayBy.DEATHS,
                percent_of_touched_counties=percent_of_deaths_counties,
                last_date=last_date,
                death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
                death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
            )

        ending_days_by_infections = \
            RealData.ending_days_by_death_toll_slope(
                starting_days_by=StartingDayBy.INFECTIONS,
                percent_of_touched_counties=percent_of_infected_counties,
                last_date=last_date,
                death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
                death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
            )

        # get list of voivodeships to iterate over them while getting pandemic duration
        starting_days_by_infections = sort_dict_by_values(
            starting_days_by_infections)
        voivodeships_synchro = starting_days_by_infections.keys()

        # prepare plotting colors and transparencies
        color_deaths = 'C0'
        color_infections = 'C1'
        alpha_deaths = 0.4
        alpha_infections = 0.4

        # Plot days when first phase of pandemic was observed
        for voivodeship in voivodeships_synchro:

            x = voivodeship

            # Plot pandemic duration by deaths
            # Get start day
            y_deaths_down = starting_days_by_deaths[voivodeship]
            # Get end day (if is np.NaN end_day=start_day)
            if np.isnan(ending_days_by_deaths[voivodeship]):
                y_deaths_up = y_deaths_down
            else:
                y_deaths_up = ending_days_by_deaths[voivodeship]
            markerline, stemlines, baseline = ax.stem(x,
                                                      y_deaths_up,
                                                      bottom=y_deaths_down,
                                                      basefmt='C0o')
            markerline.set_markersize(15)
            baseline.set_markersize(15)
            stemlines.set_linewidth(6)

            markerline.set_color(color_deaths)
            baseline.set_color(color_deaths)
            stemlines.set_color(color_deaths)

            markerline.set_alpha(alpha_deaths)
            baseline.set_alpha(alpha_deaths)
            stemlines.set_alpha(alpha_deaths)

            # plot pandemic duration by infections
            y_infections_up = ending_days_by_infections[voivodeship]
            y_infections_down = starting_days_by_infections[voivodeship]
            markerline, stemlines, baseline = ax.stem(x,
                                                      y_infections_up,
                                                      bottom=y_infections_down,
                                                      basefmt='C1o')
            markerline.set_markersize(15)
            baseline.set_markersize(15)
            stemlines.set_linewidth(6)

            markerline.set_color(color_infections)
            baseline.set_color(color_infections)
            stemlines.set_color(color_infections)

            markerline.set_alpha(alpha_infections)
            baseline.set_alpha(alpha_infections)
            stemlines.set_alpha(alpha_infections)

        # manually create legend entries
        deaths_patch = mPatches.Patch(color='None',
                                      label=f'Days found by {percent_of_deaths_counties}% '
                                            f'of counties where someone died')
        infections_patch = mPatches.Patch(color='None',
                                          label=f'Days found by {percent_of_infected_counties}% '
                                                f'of counties where someone got infected')

        # create legend entries to a legend
        leg = ax.legend(handles=[deaths_patch, infections_patch])

        # change text color of legend entries description
        colors = [color_deaths, color_infections]
        for i, (h, t) in enumerate(zip(leg.legendHandles, leg.get_texts())):
            t.set_color(colors[i])

        # remove line/marker symbol from legend (leave only colored description)
        for i in leg._legend_handle_box.get_children()[
            0].get_children():  # noqa (suppres warning)
            i.get_children()[0].set_visible(False)

        # rotate x labels (voivodeship names)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        # improve y lower limit
        ax.set_ylim([0, None])

        plt.tight_layout()
        cls.__show_and_save(fig=fig,
                            plot_type='Days of pandemic comparison',
                            plot_name=(
                                f'Counties death {percent_of_deaths_counties} percent,   '
                                f'infections {percent_of_infected_counties} percent'),
                            save=save,
                            show=show)


class SimulatedVisualisation:
    """
    Class responsible for making visualisation of real and simulated data.

    Methods:
        * __show_and_save --> responsible for showing, saving and closing figure.

        * show_real_death_toll_for_voivodeship_shifted_by_hand --> responsible for
            plotting real death toll shifted among X axis such a given day, let's
            say day 10, looks like the beginning of pandemic in given voivodeship.
            Shift was estimated by looking since how many deaths' death toll looks
            'nicely', so it is a bit unreliable. Also plots simulated death toll,
            shifted that in day 10 simulated death toll equal real death toll.

        * plot_stochastic_1D_death_toll_dynamic --> one line = one, not averaged,
            simulation. Illustrate how similar simulations goes for the same model
            parameters.

        * plot_1D_modelReporter_dynamic_parameter_sweep --> plots death toll as follows:
            - takes one of model parameter ['beta', 'mortality', 'visibility'], let say beta
            - creates one figure for each unique (mortality, visibility) pair
            - X axis = t, days;
            - creates one line for each beta. Line represents simulated values
                from one of reporters from model DataCollector.

        * max_death_toll_fig_param1_xAxis_param2_series_param3 --> plot max death toll as follows:
            - one figure for each param1 value.
            - param2 as X axis.
            - one line for each param3 value.

        * __plot_matched_real_death_toll_to_simulated --> gets y1 and y2 data arrays
            and some X length. Plots, y1, y2, and y1 shifted to best match y2 on given
            X interval. Just make plot without axis labels, only make labels
            ['simulated', 'real', 'real_shifted'].

    """

    @classmethod
    def __show_and_save(cls, fig, plot_type, plot_name, save, show,
                        dir_to_data=None, save_in_general=False):
        """Function that shows and saves figures

        :param fig: figure to be shown/saved
        :type fig: matplotlib figure
        :param dir_to_data: directory to simulated data, used in image output directory
        :type dir_to_data: None or str
        :param save_in_general: save in Result/plots/.. ?
        :type save_in_general: bool
        :param plot_type: general description of a plot type, each unique type will have its own folder
        :type plot_type: str
        :param plot_name: detailed name of plot, it will serve as filename
        :type plot_name: str
        :param save: save plot?
        :type save: bool
        :param show: show plot?
        :type show: bool
        """

        # set fig as current figure
        plt.figure(fig)
        if save:
            if dir_to_data:
                save_dir = dir_to_data.replace('raw data', 'plots')
                save_dir += plot_type + '/'

                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir + plot_name + '.pdf')

            if save_in_general:
                save_dir = Directories.ABM_DIR + '/RESULTS/plots/' + plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir + plot_name + '.pdf')

        if show:
            plt.show()

        plt.close(fig)

    @classmethod
    def _date_from_day_number(cls, day_number: int) -> str:
        date0 = datetime.datetime(2020, 3, 4)
        date0 += datetime.timedelta(days=day_number)
        return date0.strftime('%Y-%m-%d')

    @classmethod
    def _day_number_from_date(cls, date: str) -> int:
        date0 = datetime.datetime(2020, 3, 4)
        return (datetime.datetime.strptime(date, "%Y-%m-%d") - date0).days

    @classmethod
    def plot_stochastic_1D_death_toll_dynamic(cls,
                                              avg_directory: str,
                                              not_avg_directory: str,
                                              voivodeship: str,
                                              show=False,
                                              save=True):
        """
        Designed to show stochasticity of simulations. Plots many death toll lines at the same Axes, each line was
        generated by simulation with exactly the same parameters.

        :param avg_directory: directory to saved averaged model level DataCollector results. It's just for taking plot
            title name from filename (for reading simulation run parameters).
            Params are read from the latest file in directory.
        :type avg_directory: str
        :param not_avg_directory: directory to raw data from simulation, usually stored in TMP_SAVE/
        :type not_avg_directory: str
        :param voivodeship: name of voivodeship that will be included in plot title.
        :type voivodeship: str
        :param save: Save plot?
        :type save: Boolean
        :param show: Show plot?
        :type show: Boolean

        If in run.py all params are fixed and iterations > 1 then desired not averaged data are stored in TMP_SAVE.
        """

        # get not averaged data
        dict_result = Results.get_single_results(
            not_avg_data_directory=not_avg_directory)

        # get beta value from latest file in avg_directory
        variable_params = variable_params_from_fname(
            fname=latest_file_in_dir(avg_directory))
        beta = variable_params[TRANSLATE.to_short('beta')]

        # create figure, axes, titles and labels
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title(
            f"Death toll stochastic for {voivodeship} with " + r"$\beta$=" + f"{beta}")
        ax.set_xlabel('t, days')
        ax.set_ylabel(r'Death toll')

        # plot not avg data (stochastic)
        for df in dict_result.values():
            ax.plot(df['Day'], df['Dead people'])

        plt.tight_layout()

        cls.__show_and_save(fig=fig,
                            dir_to_data=avg_directory,
                            plot_type="Death toll stochastic",
                            plot_name=f"Death toll stochastic for {voivodeship} with beta={beta}",
                            save=save,
                            show=show)

    @classmethod
    def plot_1D_modelReporter_dynamic_parameter_sweep(cls,
                                                      directory: str,
                                                      model_reporter: str,
                                                      parameter: str,
                                                      normalized=False,
                                                      plot_real=True,
                                                      show=True,
                                                      save=False):
        # sourcery no-metrics
        """
        Plots averaged DataCollector model_reporter from simulations.
        One figure for one mortality-visibility pair.
        One line for each beta.


        :param directory: directory to averaged simulated data
        :type directory: str
        :param model_reporter: name of model reporter from model
            DataCollector, which data will be plotted. Allowed inputs:
            ['Dead people', 'Infected toll']
        :type model_reporter: str
        :param parameter: parametr, which will be swept (one plot line for each value)
        :type parameter: str
        :param normalized: normalize death toll to 1 in order to see how quickly it saturates?
        :type normalized: bool
        :param plot_real: also plot real data?
        :type plot_real: bool
        :param show: show plot?
        :type show: bool
        :param save: save plot?
        :type save: bool
        """

        # check if model_reporter parameter is ok
        model_reporter = model_reporter.capitalize()
        if model_reporter not in ['Dead people', 'Infected toll']:
            raise ValueError(f"In function"
                             f" plot_1D_modelReporter_dynamic_parameter_sweep "
                             f"was passed illegal argument for "
                             f"'model_reporter' param. Passed"
                             f"{model_reporter} while allowed value is one"
                             f"of the following: ['Dead people', 'Infected_toll'].")

        reporter_to_name = {
            'Dead people': 'Death toll',
            'Infected toll': 'Infected toll'
        }

        # get parameters that will make pairs of fixed values
        # one such pair for one plot
        parameter = TRANSLATE.to_short(parameter)
        possible_params = TRANSLATE.to_short(
            ['beta', 'mortality', 'visibility'])
        possible_params.remove(parameter)
        param2, param3 = possible_params

        # check if all params are ok
        check_uniqueness_and_correctness_of_params(param1=parameter,
                                                   param2=param2,
                                                   param3=param3)

        # get fnames from directory and also voivodeship name
        fnames = all_fnames_from_dir(directory=directory)
        voivodeship = voivodeship_from_fname(fname=fnames[0])

        # get fixed simulation params from any (first) filename
        fixed_params = fixed_params_from_fname(fname=fnames[0])

        """
        Group fnames such: grouped[i][j] = fname((param2, param3)_i, parameter_j).
        That is crucial because:
            each pair (param2, param3)_j creates new figure
            each parameter_i value creates one series (line)

        So it will go like this:
        for each_pair in pairs: ... create new figure ...
            for param_value in param_values:
                plot line ...
        """
        grouped_fnames = group_fnames_by_pair_param2_param3_and_param1(
            directory=directory,
            param1=parameter,
            param2=param2,
            param3=param3)

        num_of_plots, num_of_lines = grouped_fnames.shape

        # prepare beata_string which prints itself as beta symbol instead of just 'beta'
        # to use it while polot instead of use f'{param}' use f'{param.replace('beta', beta_str)}'
        beta_str = r'$\beta$'

        # for each pair create new figure
        for plot_id in range(num_of_plots):

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # make title, which includes voivodeship and normalization
            reporter_info = reporter_to_name[model_reporter]
            voivodeship_info = ''
            norm_info = ''

            if plot_real and voivodeship is not None:
                voivodeship_info = f'for {voivodeship}'
            if normalized:
                norm_info = ' (normalized)'

            # for title get info about values of params in a pair
            fname = grouped_fnames[plot_id][0]
            param2_value = variable_params_from_fname(fname=fname)[param2]
            param3_value = variable_params_from_fname(fname=fname)[param3]

            # make title of a plot
            main_title = (f"{reporter_info} {voivodeship_info} {norm_info}\n"
                          f"{param2.replace('beta', beta_str)}={float(param2_value) * 100:.1f}%    "
                          f"{param3.replace('beta', beta_str)}={float(param3_value) * 100:.1f}%\n")

            title = main_title + get_ax_title_from_fixed_params(
                fixed_params=fixed_params)
            ax.set_title(title)
            ax.set_xlabel('t, days')
            if normalized:
                ax.set_ylabel(f'{reporter_info} / max({reporter_info})')
            else:
                ax.set_ylabel(f'{reporter_info}')

            # Prepare empty array for parameter values (one value = one line).
            # Also prepare empty array for max DataCollector value for each parameter value,
            # it will be in legend if DataCollector entries will be normalized.
            parameter_values = np.empty(num_of_lines)
            model_reporter_max_values = np.empty_like(parameter_values)

            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

            # on created figure plot lines representing death toll for each beta,
            # so iterate over betas
            for line_id in range(num_of_lines):

                fname = grouped_fnames[plot_id][line_id]
                df = pd.read_csv(fname)
                variable_params = variable_params_from_fname(fname=fname)

                parameter_values[line_id] = variable_params[parameter]

                # make legend, but without [param2, param3] info because it already stands in fig title
                # also ignore 'beta' and use r'$\beta$'
                ignored = [param2, param3, 'beta']
                print(ignored)
                # legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=ignored)
                legend = f"{parameter.replace('beta', beta_str)}={float(variable_params[parameter]) * 100:.2f}%"

                # if normalized add info about not normalized value of modelReporter
                if normalized:
                    legend += ' ' * 4 + f'{reporter_info} = {np.max(df[model_reporter]):.0f}'
                    ax.plot(df['Day'],
                            df[model_reporter] / np.max(df[model_reporter]),
                            label=legend,
                            color=colors[line_id])
                else:
                    ax.plot(df['Day'], df[model_reporter], label=legend,
                            color=colors[line_id])

                model_reporter_max_values[line_id] = np.max(df[model_reporter])

            # also plot real live pandemic data if you want
            if plot_real and voivodeship is not None:
                legend = 'Real data'
                if model_reporter == 'Dead people':
                    real_data = RealData.death_tolls()
                elif model_reporter == 'Infected toll':
                    real_data = RealData.get_real_infected_toll()
                else:
                    raise ValueError(f'model_reporter {model_reporter} not '
                                     f'implemented in RealData class')

                y = np.array(real_data.loc[voivodeship].to_list())

                # Find first day for which real life data is
                # grater than the greatest simulated, to nicely truncate plot.
                last_real_day = np.argmax(y > max(model_reporter_max_values))
                if last_real_day == 0:
                    last_real_day = len(y)

                # plot initial part of real death toll
                x = range(last_real_day)
                y = y[:last_real_day]
                ax.plot(x, y, label=legend, color='black', linewidth=4,
                        alpha=0.5)

            # set legend entries in same order as lines were plotted (which was by ascending beta values)
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(
                *sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)

            plt.tight_layout()

            main_title = main_title.replace(beta_str, 'beta').replace('\n', '')
            cls.__show_and_save(fig=fig,
                                dir_to_data=directory,
                                plot_type=f'{reporter_to_name[model_reporter]} dynamic {parameter} sweep',
                                plot_name=" ".join(main_title.split()),
                                save=save,
                                show=show)

    @classmethod
    def max_death_toll_fig_param1_xAxis_param2_series_param3(cls,
                                                             directory: str,
                                                             param1: str,
                                                             param2: str,
                                                             param3: str,
                                                             show=True,
                                                             save=False):
        """
        PLots max death toll:
            one figure for one param1 value.
            param2 on x-axis.
            one line for one param3 value.

        :param directory: directory to averaged simulated data
        :type directory: str
        :param param1: 'mortality' or 'visibility' or 'beta'
        :type param1: str
        :param param2: 'mortality' or 'visibility' or 'beta', but not the same as param1
        :type param2: str
        :param param3: 'mortality' or 'visibility' or 'beta', but not the same as param1 and param2
        :type param3: str
        :param show: show plot?
        :type show: bool
        :param save: save plot?
        :type save: bool
        """

        # Convert param names to current naming convention
        param1, param2, param3 = TRANSLATE.to_short([param1, param2, param3])

        # verify if all model params passed to this function are correct
        check_uniqueness_and_correctness_of_params(param1=param1,
                                                   param2=param2,
                                                   param3=param3)

        fnames = all_fnames_from_dir(directory=directory)

        # get fixed simulation params from any (first) filename
        fixed_params = fixed_params_from_fname(fname=fnames[0])

        """
        Group fnames such: grouped[i][j][k] = fname(param1_i, param2_j, param3_k).

        Explanation why order of params 2 and 3 is swapped:
            assume that the goal is to make plots where:
                one figure for one visibility
                X axis = mortality
                one line for one beta
            then params passed o this function
            'max_death_toll_fig_param1_xAxis_param2_series_param3'
            will be like:
                param1 = 'visibility'
                param2 = 'mortality'
                param3 = 'beta'

            To plot that I need to iterate over 3 loops in a following order:
                for visibility in visibilities: ... (make figure)
                    for beta in betas: ...
                        for mortality in mortalities: ... (death[beta][mortality] = val)
                    plot(death[beta])

            So to achieve x_mortality, series_beta I need to group fnames by
            fnames[visibility][beta][mortality] == fnames[param1][param3][param2]
        """
        grouped_fnames = group_fnames_by_param1_param2_param3(
            directory=directory,
            param1=param1,
            param2=param3,
            param3=param2)

        [print(grouped_fnames[0][i][0]) for i in range(len(grouped_fnames[0]))]

        # get range for ijk (explained above)
        num_of_plots, num_of_lines, num_of_points_in_line = grouped_fnames.shape

        # set colormap (each line will have its own unique color)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

        beta_str = r'$\beta$'

        # for each param1 value create a new figure
        for plot_id in range(num_of_plots):

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # set title title
            param1_value = \
                variable_params_from_fname(
                    fname=grouped_fnames[plot_id][0][0])[
                    param1]

            main_title = (f"Death toll max for "
                          f"{param1.replace('beta', beta_str)}={float(param1_value) * 100:.0f}%  "
                          f"(after {get_last_simulated_day(fname=grouped_fnames[plot_id][0][0]) + 1} days)\n")

            title = main_title + get_ax_title_from_fixed_params(
                fixed_params=fixed_params)
            ax.set_title(title)

            # set x and y labels
            ax.set_xlabel(param2.replace('beta', beta_str))
            ax.set_ylabel(r'Death toll')

            # prepare empty array for max death toll for each line
            # deaths[i][j] -> num of deaths for i-th line at x = param3[j]
            deaths = np.zeros((num_of_lines, num_of_points_in_line))

            # prepare dict for param describing line (param3)
            params3_index_to_value = {}

            # run over lines
            for line_id in range(num_of_lines):

                # get value af parm2 which describes current line
                fname = grouped_fnames[plot_id][line_id][0]
                variable_params = variable_params_from_fname(fname=fname)
                params3_index_to_value[line_id] = variable_params[param3]

                # prepare dict for param describing x component of points on a line (param2)
                params2_index_to_value = {}

                # run over points on a line
                for point_id in range(num_of_points_in_line):
                    # get x coordinate of a point
                    fname = grouped_fnames[plot_id][line_id][point_id]
                    variable_params = variable_params_from_fname(fname=fname)
                    params2_index_to_value[point_id] = variable_params[param2]

                    df = pd.read_csv(fname)

                    # fill deaths array by death toll for current point
                    # y - death toll from file[plot_id][line_id][point_id]
                    deaths[line_id][point_id] = np.max(df['Dead people'])

                # make label for one line (fixed param2)
                label = f"{param3.replace('beta', beta_str)}={params3_index_to_value[line_id]}"

                # plot data
                ax.plot(params2_index_to_value.values(),
                        deaths[line_id],
                        label=label,
                        color=colors[line_id],
                        marker='o',
                        markersize=5)

            # set legend entries in same order as lines were plotted (which was ascending by param2 values)
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(
                *sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)

            # set lower y limit to 0
            ax.set_ylim(bottom=0)
            plt.tight_layout()

            # handles showing and saving simulation
            main_title = main_title.replace('\n', '').replace(r'$\beta$',
                                                              'beta')
            cls.__show_and_save(fig=fig,
                                dir_to_data=directory,
                                plot_type='Death toll max, x mortality, beta series',
                                plot_name=" ".join(main_title.split()),
                                save=save,
                                show=show)

    @classmethod
    def death_toll_for_best_tuned_params(cls,
                                         last_date='2020-07-01',
                                         show=True,
                                         save=True):  # sourcery no-metrics

        def plot_matched_real_death_toll_to_simulated(
                voivodeship: str,
                simulated_death_toll: np.ndarray,
                real_death_toll: Union[pd.Series, np.ndarray],
                simulated_recovered_toll: np.ndarray,
                real_recovered_toll: Union[pd.Series, np.ndarray],
                starting_day: int,
                ending_day: int,
                last_date_str: str,
        ):
            """
            Assumes y(x=0) = y[0], y(x=1) = y[1] and so on.
            Function moves y2 data along X axis to find out for what shift subset of y2 = y2[start: end]
            best matches subset of the same length of y1. Then plot y1, y2, y2_shifted.

            As it is a general function it returns `plt.figure`, so that it can be modified
            by function one level above.

            """
            # get shift
            df_best_tuned_csv_details = OptimizeResultsWriterReader.get_n_best_tuned_results(
                n=1)
            df_best_run_details = df_best_tuned_csv_details.loc[voivodeship]
            shift = int(df_best_run_details['shift'])

            # get last day for which data will be plotted
            last_day_to_search = list(RealData.death_tolls().columns).index(
                last_date_str)

            # truncate data up to given date
            real_death_toll = real_death_toll[:last_day_to_search]
            simulated_death_toll = simulated_death_toll[
                                   :last_day_to_search - shift]

            fig = plt.figure(figsize=(12, 8))

            gs = GridSpec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])

            # prepare line colors
            color_real = 'C0'
            color_simulated = 'C1'

            # Plot infected toll ------------------------------------------------------
            ax1.plot(
                range(len(real_death_toll)),
                real_death_toll,
                label='raportowana suma zgonów',
                color=color_real,
                lw=2,
                zorder=1)

            # plot shifted simulated death toll
            ax1.plot(
                np.arange(0, len(simulated_death_toll)) + shift,
                simulated_death_toll,
                label='symulowana suma zgonów',
                color=color_simulated,
                lw=5,
                alpha=0.6,
                zorder=0)

            # plot vertical lines, which will mark segment of real death toll
            # for which simulated death toll was tuned
            ax1.axvline(starting_day, color='red')
            ax1.axvline(ending_day, color='red',
                        label='dni do których dopasowywano model')

            # get max of real death toll to set y_axis limit and annotate num of fitted days
            max_death_toll = np.nanmax(real_death_toll)

            # annotate how many days was fitted (draw arrow)
            ax1.annotate(
                "",
                xy=(starting_day, max_death_toll),
                xytext=(ending_day, max_death_toll),
                arrowprops=dict(arrowstyle="<->"))

            # annotate how many days was fitted (write days over arrow)
            ax1.annotate(
                f"{ending_day - starting_day} dni",
                xy=((ending_day + starting_day) / 2, max_death_toll * 1.02),
                ha='center',
                va='bottom',
                fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.8,),
                          ec="none"),
            )

            # annotate day 0 as 2020-03-04
            ax1.set_xticks(ax1.get_xticks().tolist()[1:-1])
            ax1.set_xticklabels(
                [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
                 ax1.get_xticks().tolist()])

            # annotate starting fit date
            ax1.text(
                starting_day,
                max_death_toll / 2,
                cls._date_from_day_number(starting_day),
                rotation=90,
                ha='right',
                va='center',
                fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            # annotate ending fit date
            ax1.text(
                ending_day,
                max_death_toll / 2,
                cls._date_from_day_number(ending_day),
                rotation=90,
                ha='left',
                va='center',
                fontsize=20,
                bbox=dict(boxstyle="square,pad=0",
                          fc=mColors.to_rgba('white')[:-1] + (0.5,),
                          ec="none"),
            )

            # Set reasonable y_ax1is limits
            ax1.set_ylim([-0.1 * max_death_toll, 1.5 * max_death_toll])

            ax1.legend(loc='upper left', framealpha=1)

            # Plot recovered toll ------------------------------------------------------
            # plot real recovered toll

            real_recovered_toll = real_recovered_toll[:last_day_to_search]
            simulated_recovered_toll = simulated_recovered_toll[
                                       :last_day_to_search - shift]

            ax2.plot(
                range(len(real_recovered_toll)),
                real_recovered_toll,
                label='raportowana suma wyzdrowień',
                color=color_real,
                lw=2,
                zorder=1)

            # plot shifted simulated recovered toll
            ax2.plot(
                np.arange(0, len(simulated_recovered_toll)) + shift,
                simulated_recovered_toll,
                label='symulowana suma wyzdrowień',
                color=color_simulated,
                lw=2,
                alpha=0.6,
                zorder=0)

            y_annotate_offset = .1 * max(simulated_recovered_toll)

            ax2.annotate(
                f"{real_recovered_toll[ending_day] / 1000:.1f}k",
                xy=(ending_day, real_recovered_toll[ending_day]),
                xytext=(ending_day - 10, real_recovered_toll[
                    ending_day] + .5 * y_annotate_offset),
                color=color_real,
                fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none",
                          lw=2),
            )

            ax2.annotate(
                f"{simulated_recovered_toll[ending_day - shift] / 1000:.1f}k",
                xy=(ending_day, simulated_recovered_toll[ending_day - shift]),
                xytext=(ending_day - 10, simulated_recovered_toll[
                    ending_day - shift] + 2 * y_annotate_offset),
                color=color_simulated,
                fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none",
                          lw=2))

            def format_number(*args):
                """To write numbers of recovered with 'k' suffix. 1500 = 1.5k"""

                value = args[0]
                return (
                    '{:1.0f}k'.format(value * 0.001)
                    if value != 0
                    else '{:0.0f}'.format(value)
                )

            ax2.yaxis.set_major_formatter(format_number)
            ax2.axvline(ending_day, color='red')

            # annotate day 0 as 2020-03-04
            ax2.set_xticks(ax2.get_xticks().tolist()[1:-1])
            ax2.set_xticklabels(
                [f'{x:.0f}' if x != 0 else '2020-03-04' for x in
                 ax2.get_xticks().tolist()])

            ax2.legend(loc='upper left')

            return ax1, ax2, fig

        def make_plot(sub_df):
            """
            Call 'plot_matched_real_death_toll_to_simulated' and add
            new details to plot returned by it.
            """

            def label_ax(ax: plt.axes, title: str, xaxis_label: str,
                         yaxis_label) -> None:
                ax.set_title(title)
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel(yaxis_label)

            # get real death toll
            death_tolls = RealData.death_tolls()
            recovered_tolls = RealData.recovered_tolls()

            # from summary df_one_voivodeship_only get row with best tuned params
            best_run = sub_df.iloc[0]

            # read summary details from that row
            voivodeship = best_run['voivodeship']
            starting_day = int(best_run['first day'])
            ending_day = int(best_run['last day'])
            visibility = best_run['visibility']
            mortality = best_run['mortality']
            beta = best_run['beta']
            fname = best_run['fname']

            # read avg_df which contains course of pandemic in simulation
            avg_df = pd.read_csv(fname)

            # make plot (only with data labels; no axis, titles etc.)
            ax1, ax2, fig = plot_matched_real_death_toll_to_simulated(
                voivodeship=voivodeship,
                real_death_toll=death_tolls.loc[voivodeship],
                simulated_death_toll=avg_df['Dead people'],
                real_recovered_toll=recovered_tolls.loc[voivodeship],
                simulated_recovered_toll=avg_df['Recovery people'],
                starting_day=starting_day,
                ending_day=ending_day,
                last_date_str=last_date,
            )

            # create title and axis labels
            fig.suptitle(
                f'{voivodeship.capitalize()}, rezultat dopasowania parametrów modelu.\n'
                r"$\beta$"f'={beta * 100:.2f}%, '
                f'śmiertelność={mortality:.2f}%, '
                f'widoczność={visibility * 100:.0f}%')
            label_ax(ax1, 'Zgony', 't, dni', 'Suma zgonów')
            label_ax(ax2, 'Wyzdrowienia', 't, dni', 'Suma wyzdrowień')

            # Hide the right and top spines
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            plt.tight_layout()

            # SAVING ------------------------------------------------------------------------------
            # handles showing and saving simulation
            main_title = f"{voivodeship} finest model params tuning"
            # save in general plots folder
            if save:
                plot_type = 'Best death toll fits, beta and mortality'
                save_dir = Directories.ABM_DIR + '/RESULTS/plots/' + plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plot_name = " ".join(main_title.split())

                plt.savefig(save_dir + plot_name + '.png')

            # show and save in avg_simulation folder
            cls.__show_and_save(fig=fig,
                                dir_to_data=os.path.split(fname)[0] + '/',
                                plot_type='Auto fit by percent_of_touched_counties',
                                plot_name=" ".join(main_title.split()),
                                save=save,
                                show=show)

        # get summary df
        df_tuning = OptimizeResultsWriterReader.get_all_tuning_results()

        # make plot for each voivodeship, which was tuned
        for voivodeship_tuned in df_tuning['voivodeship'].unique():
            make_plot(sub_df=df_tuning.loc[
                df_tuning['voivodeship'] == voivodeship_tuned])

    @classmethod
    def plot_best_beta_mortality_pairs(cls,
                                       pairs_per_voivodeship: int,
                                       show=True,
                                       save=False):

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        fig.suptitle(
            "Najlepiej dopasowane wartości "r'$\beta$'" i śmiertelności"
            " dla poszczególnych województw.")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Śmiertelność")

        best_df = OptimizeResultsWriterReader.get_n_best_tuned_results(
            n=pairs_per_voivodeship)

        voivodeships = best_df['voivodeship']
        x = best_df['beta']
        y = best_df['mortality']

        # group voivodeships by duration of first phase
        num_of_days = best_df['last day'] - best_df['first day']

        palette = plt.get_cmap('Blues')

        ax = sns.scatterplot(
            data=best_df,
            x='beta',
            y='mortality',
            hue=num_of_days,
            style=best_df['fit error per day'] < 40,
            style_order=[True, False],
            s=500,
            palette=palette,
            edgecolor="black",
            lw=4,
            legend=False,
        )

        # Create colorbar contain pandemic duration
        norm = mColors.Normalize(vmin=min(num_of_days),
                                 vmax=max(num_of_days))

        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array(np.array([]))
        ax.figure.colorbar(sm, label='Ilość dni do których dopasowywano model')

        good_line = mLines.Line2D([], [], color='grey', marker='o', mec='k',
                                  markersize=20, ls='None',
                                  label='Dobre dopasowanie')

        bad_line = mLines.Line2D([], [], color='grey', marker='X', mec='k',
                                 markersize=20, ls='None',
                                 label='Złe dopasowanie')

        ax.legend(handles=[good_line, bad_line],
                  labelspacing=2,
                  borderpad=1.2,
                  loc='lower right')

        texts = [ax.text(x_pos, y_pos, voivodeship,
                         ha='center', va='center',
                         bbox=dict(boxstyle="square,pad=0.3",
                                   fc=mColors.to_rgba('lightgrey')[:-1] + (1,),
                                   ec="none"),
                         ) for x_pos, y_pos, voivodeship in
                 zip(x, y, voivodeships)]
        adjust_text(
            texts,
            expand_points=(1.5, 1.5),
        )

        cls.__show_and_save(
            fig=fig,
            plot_type='Best beta mortality pairs',
            plot_name=f'{pairs_per_voivodeship} pairs per voivodeship',
            save=save,
            show=show,
            dir_to_data=None,
            save_in_general=True,
        )

    @classmethod
    def all_death_toll_from_dir_by_fixed_params(cls, fixed_params: dict,
                                                c_norm_type='log'):
        """Plot simple death toll dynamic to ensure that params in model works fine.

        fixed_params: dict defining folder in ABM/RESULTS, which will be plotted.

        Note: fnames should contain exactly 4 params, where 3 of them should be
            beta, mortality, visibility. Example fname:
            'Id=0001__p_inf_cust=2.34__b=0.025__m=0.02__v=0.65.csv'
        """

        def get_param_name_from_fname(fname):
            """Returns first param name found in fname except from: 'beta', 'mortality', 'visibility'."""

            fname = os.path.split(fname)[1]  # get fname without prev dirs
            fname_splitted = fname_to_list(
                fname)  # convert fname to list like ['Beta=0.036', ...]

            for param in fname_splitted:
                if (
                        'beta' not in param.lower()
                        and 'mortality' not in param.lower()
                        and 'visibility' not in param.lower()
                ):
                    return param.split('=')[0].lower()

        def sort_helper(fname, parameter_name):
            fname = os.path.split(fname)[1]  # get fname without prev dirs
            fname_splitted = fname_to_list(
                fname)  # convert fname to list like ['Beta=0.036', ...]

            # make dict {'beta': 0.036, ...}
            fname_dict = {item.split('=')[0].lower(): float(item.split('=')[1])
                          for item in fname_splitted}
            return fname_dict[parameter_name.lower()]

        # Get folder name (which contains csv files)
        folder = find_folder_by_fixed_params(
            directory=Directories.AVG_SAVE_DIR,
            params=fixed_params)
        folder += 'raw data/'

        # Get fnames and sort them by param values
        fnames = all_fnames_from_dir(directory=folder)
        param_name = get_param_name_from_fname(fnames[0])
        fnames = sorted(fnames,
                        key=lambda fname: sort_helper(fname, param_name))
        param_name = TRANSLATE.to_short(param_name)

        # Get variable params from fnames and dfs from its csv files
        variable_params = [
            TRANSLATE.to_short(variable_params_from_fname(fname)) for fname in
            fnames]
        dfs = [pd.read_csv(fname) for fname in fnames]

        # Prepare  fig and ax
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        sns.despine(fig=fig, ax=ax)

        fig.suptitle("Suma zgonów")
        ax.set_title(f"Różne wartości '{param_name.replace('_', ' ')}'")

        # Create rainbow colormap and colors; based on param values
        # get min and max param values
        param_values = np.array(
            [float(params[param_name]) for params in variable_params])
        min_param = np.min(param_values[np.nonzero(param_values)])
        max_param = np.max(param_values[np.nonzero(param_values)])

        cmap = matplotlib.cm.get_cmap('rainbow')
        # set linear or logarithm colormap
        if c_norm_type == 'log':
            # protection in case when 'min(param_value) == 0'
            if 0 not in param_values:
                log_norm = mColors.LogNorm(vmin=min_param, vmax=max_param)
                colors = cmap(log_norm(param_values))
            else:
                log_norm = mColors.LogNorm(vmin=min_param,
                                           vmax=max_param + min_param)
                colors = cmap(log_norm(param_values + min_param))
        else:
            lin_norm = mColors.Normalize(vmin=min_param, vmax=max_param)
            colors = cmap(lin_norm(param_values))

        # PLot data
        for params, df, color in zip(variable_params, dfs, colors):
            ax.plot(df['Day'], df['Dead people'],
                    label=params[param_name], c=color)

        # label lines by putting text on top of them
        labelLines(ax.get_lines(), zorder=2)
        plt.show()


def created():
    VisRealDT(last_day_to_plot=200).plot()
    VisRealDTShiftedByHand().plot()
    VisSummedRealDT(last_day_to_plot=180).plot()
    VisSummedRealI(last_day_to_plot=200).plot()
    VisSummedRealD(last_day_to_plot=200).plot()


def main():
    # VisBestTunedDTDynamic().plot()
    # SimulatedVisualisation.death_toll_for_best_tuned_params(
    # show=True, save=False)

    # SimulatedVisualisation.plot_1D_modelReporter_dynamic_parameter_sweep(
    #     directory='C:/Users/HAL/PycharmProjects/ABM/RESULTS/raw data/Runs=12__g_size=(30, 30)__h_size=3__n=800',
    #     model_reporter='Dead people',
    #     parameter='mortality',
    #     show=True, save=False,
    # )

    RealVisualisation.plot_last_day_finding_process(
        voivodeships=('all',),
        start_days_by=StartingDayBy.INFECTIONS,
        percent_of_touched_counties=80,
        save=True,
        show=False
    )


if __name__ == '__main__':
    main()
