"""Making all kind of plots to visualize real, simulated or both data."""
import datetime

from abc import ABC, abstractmethod
from typing import Optional

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mColors
import seaborn as sns

from adjustText import adjust_text
from labellines import labelLines

from scipy.signal import savgol_filter
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.real_data import RealData
# from disease_spread_model.model.model_adjustment import TuningModelParams

from disease_spread_model.model.my_math_utils import *
from disease_spread_model.config import Directories


# from disease_spread_model.dirs_to_plot import FolderParams

class Visualisation(ABC):
    BASE_SAVE_DIR = f'{Directories.ABM_DIR}/RESULTS/plots/'
    
    def __init__(self,
                 show: Optional[bool] = True,
                 save: Optional[bool] = True,
                 ):
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
    
    def _activate_fig(self):
        if not self.fig_activated:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111)
    
    def _show_fig(self) -> None:
        if self.show:
            plt.figure(self.fig)
            plt.show()
    
    def _save_fig(self, file_extension: Optional[str] = 'pdf') -> None:
        """Save plot."""
        if self.save:
            self._set_plot_folder_name()
            self._set_fname()
            
            plt.figure(self.fig)
            
            save_dir = self.BASE_SAVE_DIR + self.save_folder_name + '/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + self.fname + '.' + file_extension.lower())
    
    @abstractmethod
    def _label_plot_components(self) -> None:
        pass
    
    @abstractmethod
    def _set_plot_folder_name(self) -> None:
        pass
    
    @abstractmethod
    def _make_basic_plot(self) -> None:
        pass
    
    @abstractmethod
    def _make_plot_nice_looking(self) -> None:
        pass
    
    @abstractmethod
    def _make_annotations(self) -> None:
        pass
    
    @abstractmethod
    def _set_fname(self) -> None:
        pass
    
    def plot(self) -> None:
        self._make_basic_plot()
        self._make_plot_nice_looking()
        self._make_annotations()
        
        self._show_fig()
        self._save_fig()


class VisRealDT(Visualisation):
    def __init__(
            self,
            show: Optional[bool] = True,
            save: Optional[bool] = True,
            
            last_day_to_plot: Optional[int] = None,
            normalized: Optional[bool] = False,
            voivodeships: Optional[tuple[str]] = ('all',),
    ):
        super().__init__(show, save)
        self.last_day_to_plot = self._get_last_day_to_plot_proper_value(last_day_to_plot)
        self.normalized = normalized
        self.voivodeships = voivodeships
    
    @staticmethod
    def _get_last_day_to_plot_proper_value(last_day_to_plot: Optional[int]) -> int:
        if last_day_to_plot:
            return min(last_day_to_plot, len(RealData.get_real_death_toll().columns))
        else:
            return len(RealData.get_real_death_toll().columns)
    
    def _update_axis_limits(self) -> None:
        """Set nice looking axis limits."""
        
        if self.normalized:
            self.ax.set_xlim(0, self.last_day_to_plot + 5)
            self.ax.set_ylim(0, 1.1)
    
    def _get_num_of_voivodeships(self) -> int:
        if 'all' in self.voivodeships:
            return len(RealData.get_voivodeships())
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
        self.ax.legend()
        
        self.ax.set_title(f"Suma zgonów{' (znormalizowana)' if self.normalized else ''} "
                          f"od pierwszego przypadku tj. od 04-03-2020.\n"
                          f"Kolory linii oddają wsp. urbanizacji (czerwony - najniższa wartość).")
        
        if self.normalized:
            self.ax.set_ylabel("Suma zgonów / max(Suma zgonów)")
        else:
            self.ax.set_ylabel("Suma zgonów")
        
        self.ax.set_xlabel("t, dni od pierwszego śmiertelnego przypadku.")
    
    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        self._update_axis_limits()
        plt.tight_layout()
    
    def _make_annotations(self) -> None:
        
        # annotate day 0 as 2020-03-04
        self.ax.set_xticks(self.ax.get_xticks().tolist()[1:-1])
        self.ax.set_xticklabels([f'{x:.0f}' if x != 0 else '2020-03-04' for x in self.ax.get_xticks().tolist()])
    
    def _make_basic_plot(self) -> None:
        
        # get sorted urbanization, it will be criteria by which line colors are matched
        urbanization = RealData.get_real_general_data()['urbanization']
        urbanization = urbanization.sort_values()
        
        real_death_toll = RealData.get_real_death_toll()
        
        for voivodeship, color in zip(urbanization.keys(), self._get_colors()):
            # x, y data to plot
            x = list(range(self.last_day_to_plot))
            y = real_death_toll.loc[voivodeship].iloc[:self.last_day_to_plot]
            
            # normalize y data if possible and needed
            if self.normalized and np.max(y[:self.last_day_to_plot]) > 0:
                y /= np.max(y[: self.last_day_to_plot])
            
            # if current voivodeship was passed as arg to function then plot data for it
            if 'all' in self.voivodeships or voivodeship in self.voivodeships:
                self.ax.plot(x[: self.last_day_to_plot], y[: self.last_day_to_plot],
                             label=voivodeship, color=color)


class VisRealDTShiftedByHand(Visualisation):
    
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
        self.save_folder_name = 'Death toll dynamic shifted real'
    
    def _set_fname(self) -> None:
        self.fname = "Real shifted DT"
    
    # TODO liczba zgonów w tytule
    def _label_plot_components(self) -> None:
        self.fig.suptitle(self.fig_title)
        
        self.ax.set_title(f"Suma zgonów w województwach, w których przebieg pandemii, "
                          f"od {999} przypadków był podobny.\n")

        self.ax.set_xlabel(f"t, nr. dnia począwszy od dnia w którym zmarło {999} osób w danym województwie")
        self.ax.set_ylabel("Suma zgonów")
    
    def _make_plot_nice_looking(self) -> None:
        self._label_plot_components()
        plt.tight_layout()
    
    def _make_annotations(self) -> None:
        pass
        
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
        last_day = 100  # TODO replace it by dynamic value

        for init_deaths, voivodeships in death_shifts.items():
            death_toll_shifted_df = \
                RealData.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(
                    starting_day=self.starting_day,
                    minimum_deaths=init_deaths)
            
            self._make_inner_plot(voivodeships, death_toll_shifted_df, last_day)

            if self.directory_to_data:
                fnames = all_fnames_from_dir(directory=self.directory_to_data)

                num_of_lines = len(fnames)
                cmap = plt.get_cmap('viridis')
                colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

                for i, fname in enumerate(fnames):
                    beta = float(variable_params_from_fname(fname=fname)['beta'])
                    mortality = float(variable_params_from_fname(fname=fname)['mortality'])
                    visibility = float(variable_params_from_fname(fname=fname)['visibility'])

                    df = pd.read_csv(fname)

                    y = np.array(df['Dead people'])
                    try:
                        common_day = np.where(y >= init_deaths)[0][0]
                    except IndexError:
                        common_day = 0
                    y = y[common_day - self.starting_day:]
                    x = list(range(len(y)))

                    beta_info = r'$\beta$=' + f'{beta}'
                    mortality_info = f'mortality={mortality * 100:.1f}%'
                    visibility_info = f'visibility={visibility * 100:.0f}%'
                    day_info = f"day {self.starting_day}=0"

                    label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                                 mortality_info,
                                                                 visibility_info,
                                                                 day_info)

                    self.ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
                            linewidth=1, linestyle='dashed')

            self.ax.legend(prop={'family': 'monospace'}, loc='upper left')
            
            self._show_fig()
            self.fig_activated = False
            
    def _make_inner_plot(self,
                         voivodeships: list[str],
                         death_toll_shifted_df: pd.DataFrame,
                         last_day: int,
                         ) -> None:
        
        self._activate_fig()
        
        for voivodeship, color in zip(voivodeships, self._get_colors(voivodeships)):
            x = death_toll_shifted_df.columns  # x = days of pandemic = [0, 1, ...]
            y = death_toll_shifted_df.loc[voivodeship]
        
            self.ax.plot(x[:last_day], y[:last_day], c=color, lw=3, label=voivodeship)
    
    @staticmethod
    def _get_dict_shift_to_voivodeships() -> dict[int: list[str]]:
        """Return dict[starting_deaths_num] = list(voivodeship1, voivodeship2, ...)."""
        
        voivodeship_starting_deaths_dict = RealData.get_starting_deaths_by_hand()
        unique_death_shifts = sorted(list(set(voivodeship_starting_deaths_dict.values())))
        
        return {
            unique_death_shift:
                [v for v, shift in voivodeship_starting_deaths_dict.items()
                 if shift == unique_death_shift]
            for unique_death_shift in unique_death_shifts
        }

    @staticmethod
    def _get_colors(voivodeships: list) -> list:
        num_of_lines = len(voivodeships)
        cmap = plt.get_cmap('rainbow_r')
        return [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
    def plot(self) -> None:
        self._make_basic_plot()


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
            Y1 axis - day number since 04-03-2020 which is considered as first day of
                pandemic in given voivodeship, based od percentage of counties in which
                died at least one pearson
            Y2 axis - death toll in day given by Y1

            Series:
                + day number since 04-03-2020 which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    died at least one pearson.
                + day number since 04-03-2020 which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    died at least one pearson.
                + day number since 04-03-2020 which is considered as first day of
                    pandemic in given voivodeship, based od percentage of counties in which
                    at least one person fell ill.
                + death tolls correlated to both dates.


    """
    
    @classmethod
    def __show_and_save(cls, fig, plot_type, plot_name, save, show, file_format='pdf'):
        """Function that shows and saves figures

        :param fig: figure to be shown/saved
        :type fig: matplotlib figure
        :param plot_type: general description of a plot type, each unique type will have it's own folder
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

        :param title: plot title as it is seen by plt.show()
        :type title: str
        :return: plot_name: plot name which can be used as plot filename
        :rtype: str
        """
        
        plot_name = title
        plot_name = plot_name.replace('\n', ' ')
        plot_name = plot_name.replace('    ', ' ')
        if plot_name[-1:] == ' ':
            plot_name = plot_name[:-1]
        
        return plot_name
    
    @classmethod
    def show_real_death_toll(cls,
                             normalized=False,
                             voivodeships=None,
                             last_day=None,
                             show=True,
                             save=False):
        """
        Function plots normalized death toll since first day when data were collected up to last_day.
        Death toll for all voivodeships is normalized to one to show when pandemic started
        for each voivodeship. Line color gradient corresponds to urbanization (red = biggest value)

        :param normalized: normalize death toll for each voivodeship to 1?
        :type normalized: boll
        :param voivodeships: for which voivodeships plot data, if None or ['all']  plot for all
        :type voivodeships: list of str
        :param last_day: for how many days plot data, if None plot for all days
        :type last_day: int
        :param save: save plot?
        :type save: bool
        :param show: show plot?
        :type show: bool
        """
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # determine how many days will be plotted
        if last_day:
            last_day = min(last_day, len(RealData.get_real_death_toll().columns))
        else:
            last_day = len(RealData.get_real_death_toll().columns)
        
        norm_info = ' (normalized)' if normalized else ''
        ax.set_title(f"Death toll{norm_info}. Data for {last_day} days since first case i.e. 04-03-2020.\n"
                     f"The colors of the lines are matched with regard to the urbanization factor "
                     f"(red for the biggest value).")
        if normalized:
            ax.set_ylabel(f"Death toll / Day toll(day={last_day})")
        else:
            ax.set_ylabel("Death toll")
        ax.set_xlabel("t, day since first day of collecting data i.e. 04-03-2020")
        
        # set colormap
        cmap = plt.get_cmap('rainbow')
        # set proper number of colors equal to number of lines
        if 'all' in voivodeships:
            num_of_lines = len(RealData.get_voivodeships())
        else:
            num_of_lines = len(voivodeships)
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        # get sorted urbanization, it will be criteria by which line colors are matched
        urbanization = RealData.get_real_general_data()['urbanization']
        urbanization = urbanization.sort_values()
        
        real_death_toll = RealData.get_real_death_toll()
        index = 0
        # iterate over all voivodeships in order of their urbanization
        for voivodeship in urbanization.keys():
            # x, y data to plot
            x = list(range(last_day))
            y = real_death_toll.loc[voivodeship].iloc[:last_day]
            
            # normalize y data if possible
            if normalized and np.max(y[:last_day]) > 0:
                y /= np.max(y[: last_day])
            
            # if current voivodeship was passed as arg to function then plot data for it
            if 'all' in voivodeships or voivodeship in voivodeships:
                ax.plot(x[: last_day], y[: last_day], label=voivodeship, color=colors[index])
                index += 1
        
        if normalized:
            # set nice looking axis limits
            ax.set_xlim(0, last_day + 5)
            ax.set_ylim(0, 1.1)
        
        # final preparation before plotting
        ax.legend()
        plt.tight_layout()
        
        cls.__show_and_save(
            fig=fig,
            plot_type='Real death toll, since 04-03-2020',
            plot_name=cls._get_plot_name_from_title(ax.get_title()),
            save=save,
            show=show,
        )
    
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
        
        # make dict: dict[starting_deaths] = list(voivodeship1, voivodeship2, ...) *****************************
        voivodeship_starting_deaths = RealData.get_starting_deaths_by_hand()
        unique_death_shifts = sorted(list(set(voivodeship_starting_deaths.values())))
        death_shifts = {}
        for death_shift in unique_death_shifts:
            death_shifts[death_shift] = []
            for voivodeship, val in voivodeship_starting_deaths.items():
                if val == death_shift:
                    (death_shifts[death_shift]).append(voivodeship)
        # ****************************************************************************************************
        
        # for each pair (minimum_deaths, [voivodeship1, voivodeship2, ..]
        for minimum_deaths, voivodeships in death_shifts.items():
            shifted_real_death_toll = \
                RealData.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(
                    starting_day=starting_day,
                    minimum_deaths=minimum_deaths)
            
            true_start_day = RealData.get_day_of_first_n_death(n=minimum_deaths)
            
            # df_indices_order[voivodeship] = num of voivodeship sorted shifted_real_death_toll
            # by death toll in day = 60. Used to determine colors of lines in the plot.
            df_indices_order = sort_df_indices_by_col(df=shifted_real_death_toll, column=day_in_which_colors_are_set)
            death_toll_final_order = {}
            i = 0
            for voivodeship in df_indices_order:
                if voivodeship in voivodeships:
                    death_toll_final_order[voivodeship] = i
                    i += 1
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.set_title(f"Death toll for ,,similar'' voivodeships, shifted in such a way, that in day {starting_day} "
                         f"death toll is not less than {minimum_deaths}.\n"
                         f"Mapping: (voivodeship, line color) was performed in day {day_in_which_colors_are_set} "
                         f"with respect to death toll in that day.")
            ax.set_xlabel(f't, days since first {minimum_deaths} people died in given voivodeship')
            ax.set_ylabel(f'Death toll (since first {minimum_deaths} people died in given voivodeship)')
            
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
                    beta = float(variable_params_from_fname(fname=fname)['beta'])
                    mortality = float(variable_params_from_fname(fname=fname)['mortality'])
                    visibility = float(variable_params_from_fname(fname=fname)['visibility'])
                    
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
                    
                    ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
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
        ax.set_ylabel('Day number since 04-03-2020 which is considered to be the beginning of a pandemic')
        
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
                     [starting_days_infections[voivodeship] for voivodeship in voivodeships_synchro],
                     color=color_infections, linestyle='-.', marker='o', mec='black',
                     label=(f'Starting day by {percent_of_infected_counties}% of counties with '
                            f'at least one infected case.'))
        
        # plot starting_days_deaths
        l2 = ax.plot(voivodeships_synchro,
                     [starting_days_deaths[voivodeship] for voivodeship in voivodeships_synchro],
                     color=color_deaths, linestyle='-.', marker='o', mec='black',
                     label=(f'Starting day by {percent_of_death_counties}% of counties with '
                            f'at least one death case.'))
        
        # set y_lim to 0
        ax.set_ylim([-5, None])
        
        # rotate label of outer x axis
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        # **************************************************************************************************
        
        # Plot second plot on the other y axis (death toll in starting day) ********************************
        # get starting death toll by starting day by percent of touched counties
        starting_death_toll_deaths = RealData.get_starting_death_toll_for_voivodeships_by_days(
            voivodeships_days=starting_days_deaths)
        
        starting_death_toll_infections = RealData.get_starting_death_toll_for_voivodeships_by_days(
            voivodeships_days=starting_days_infections)
        
        # second y axis
        ax2 = ax.twinx()
        
        lab_death_toll_deaths = 'Death toll in starting day (by deaths).'
        lab_death_toll_infections = 'Death toll in starting day (by infections).'
        
        # plot death toll on the second y axis (normalized or not)
        if normalize_by_population:
            y_label2 = '(Death toll / population) ' r'$\cdot 10^5$'
            population = RealData.get_real_general_data()['population']
            # preserve order of voivodeship on X axis
            
            p3 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_infections[voivodeship] / population[voivodeship] * (10 ** 5)
                              for voivodeship in voivodeships_synchro],
                             color=color_infections,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_infections)
            
            p4 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_deaths[voivodeship] / population[voivodeship] * (10 ** 5)
                              for voivodeship in voivodeships_synchro],
                             color=color_deaths,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_deaths)
        else:
            y_label2 = 'Death toll (in given day)'
            
            p3 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_infections[voivodeship] for voivodeship in voivodeships_synchro],
                             color=color_infections,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_infections)
            
            p4 = ax2.scatter(voivodeships_synchro,
                             [starting_death_toll_deaths[voivodeship] for voivodeship in voivodeships_synchro],
                             color=color_deaths,
                             marker='s',
                             edgecolors='black',
                             label=lab_death_toll_deaths)
        
        # set y2 axis label
        ax2.set_ylabel(y_label2)
        # ***********************************************************************************************************
        
        # add legend (both y axis have common legend)
        plots = [l1, l2, p3, p4]
        # for some reason scatter plot is a standard object, but line plot is a list containing lines
        plots = [p if type(p) != list else p[0] for p in plots]
        labs = [p.get_label() for p in plots]
        ax.legend(plots, labs)
        plt.tight_layout()
        
        cls.__show_and_save(fig=fig,
                            plot_type='Starting days for voivodeships based on touched district',
                            plot_name=(f'percent_of_death_counties={percent_of_death_counties},   '
                                       f'percent_of_infected_counties={percent_of_infected_counties},   '
                                       f'normalize_by_population={normalize_by_population}'),
                            save=save,
                            show=show)
    
    @classmethod
    def plot_last_day_finding_process(cls,
                                      voivodeships: list[str],
                                      start_days_by='deaths',
                                      percent_of_touched_counties=20,
                                      last_date='2020-07-01',
                                      death_toll_smooth_out_win_size=21,
                                      death_toll_smooth_out_polyorder=3,
                                      derivative_half_win_size=3,
                                      plot_redundant=False,
                                      show=True,
                                      save=False,
                                      ):  # sourcery no-metrics
        """
        Plots crucial steps in finding last day of pandemic in voivodeships.

        Plots:
         - death tool
         - death tool smoothed
         - derivative of death toll
         - derivative of smoothed up death toll
         - smoothed up derivative of smoothed up death toll
         - slope of death toll
         - slope of smoothed up death toll
        """
        
        # get voivodeships
        if 'all' in voivodeships:
            voivodeships = RealData.get_voivodeships()
        
        # get start days dict
        start_days = RealData.starting_days(
            by=start_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)
        
        # get real death toll for all voivodeships (since 03.04.2019)
        death_tolls = RealData.get_real_death_toll()
        
        # fill gaps in real death toll
        for voivodeship in RealData.get_voivodeships():
            death_tolls.loc[voivodeship] = complete_missing_data(values=death_tolls.loc[voivodeship])
        
        # smooth out death toll
        death_tolls_smooth = death_tolls.copy()
        for voivodeship in RealData.get_voivodeships():
            death_tolls_smooth.loc[voivodeship] = savgol_filter(
                x=death_tolls.loc[voivodeship],
                window_length=death_toll_smooth_out_win_size,
                polyorder=death_toll_smooth_out_polyorder)
        
        # get last day in which death pandemic last day will be looked for
        last_day_to_search = list(death_tolls.columns).index(last_date)
        
        # Make plots
        for voivodeship in voivodeships:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax2 = ax.twinx()
            
            # set title and axis labels
            date0 = (datetime.datetime(2020, 3, 4) +
                     datetime.timedelta(days=int(start_days[voivodeship])))
            ax.set_title(f'Woj. {voivodeship}, szukanie końca pierwszego etapu pandemii.\n'
                         f'Dzień 0 = {date0.strftime("%Y-%m-%d")}, na podstawie '
                         f'stwierdzonego '
                         f'{"zgonu" if start_days_by == "deaths" else "zachorowania"} w '
                         f'{percent_of_touched_counties}% powiatów.\n'
                         f'Ostatni dzień - {last_date}.')
            ax.set_xlabel('t, dni')
            ax.set_ylabel('Suma zgonów')
            ax2.set_ylabel('Przeskalowana pochodna lub nachylenie sumy zgonów')
            
            day0 = start_days[voivodeship]
            x_days = np.arange(-10, last_day_to_search - day0)
            
            # death toll
            death_toll = death_tolls.loc[voivodeship]
            ax.plot(x_days, death_toll[day0 - 10: last_day_to_search],
                    color='C0', label='suma zgonów', lw=4, alpha=0.7)
            
            # death toll smoothed up
            death_toll_smooth = death_tolls_smooth.loc[voivodeship]
            ax.plot(x_days, death_toll_smooth[day0 - 10: last_day_to_search],
                    color='C1', label='suma zgonów po wygładzeniu')
            
            if plot_redundant:
                # derivative
                derivative = window_derivative(
                    y=death_toll,
                    half_win_size=derivative_half_win_size)
                ax2.plot(derivative[day0 - 10: last_day_to_search],
                         color='C2', label='pochodna sumy zgonów', alpha=0.5)
                
                # smoothed up derivative
                derivative_smoothed_up = savgol_filter(
                    x=derivative,
                    window_length=death_toll_smooth_out_win_size,
                    polyorder=death_toll_smooth_out_polyorder)
                ax2.plot(derivative_smoothed_up[day0 - 10: last_day_to_search],
                         color='black', label='wygładzona pochodna sumy zgonów',
                         alpha=1, lw=10)
                
                # derivative of smooth death toll
                derivative_smooth = window_derivative(
                    y=death_toll_smooth,
                    half_win_size=derivative_half_win_size)
                ax2.plot(derivative_smooth[day0 - 10: last_day_to_search],
                         color='C3', lw=2,
                         label='pochodna wygładzonej sumy zgonów')
                
                # smoothed up derivative of smooth death toll
                derivative_smooth_smoothed_up = savgol_filter(
                    x=derivative_smooth,
                    window_length=death_toll_smooth_out_win_size,
                    polyorder=death_toll_smooth_out_polyorder)
                ax2.plot(derivative_smooth_smoothed_up[day0 - 10: last_day_to_search],
                         color='yellow', lw=4,
                         label='wygładzona pochodna wygładzonej sumy zgonów')
                
                # slope
                slope = slope_from_linear_fit(data=death_toll, half_win_size=3)
                ax2.plot(slope[day0 - 10: last_day_to_search],
                         color='C5', alpha=0.5,
                         label='nachylenie prostej dopasowanej do'
                               ' fragmentów sumy zgonów')
            
            # slope of smooth death toll
            slope_smooth = slope_from_linear_fit(
                data=death_toll_smooth, half_win_size=3)
            
            # normalize slope to 1
            if max(slope_smooth[day0 - 10: last_day_to_search]) > 0:
                slope_smooth /= max(slope_smooth[day0 - 10: last_day_to_search])
            
            # plot normalized slope
            ax2.plot(slope_smooth[day0 - 10: last_day_to_search],
                     color='green', lw=4,
                     label='nachylenie prostej dopasowanej do'
                           ' fragmentów wygładzonej sumy zgonów')
            
            # Plot peaks (minima and maxima) of slope of smoothed up death toll.
            vec = np.copy(slope_smooth[day0 - 10: last_day_to_search])
            
            # Maxima of slope
            x_peaks_max = argrelmax(data=vec, order=8)[0]
            ax2.scatter(x_peaks_max, vec[x_peaks_max],
                        color='lime', s=140, zorder=100,
                        label='maksima nachylenia sumy zgonów')
            # Minima of slope
            x_peaks_min = argrelmin(data=vec, order=8)[0]
            ax2.scatter(x_peaks_min, vec[x_peaks_min],
                        color='darkgreen', s=140, zorder=100,
                        label='minima nachylenia sumy zgonów')
            
            # last of of pandemic as day of peak closest to 2 moths with derivative > 0.5
            # get list of candidates for last day
            last_day_candidates = [x for x in x_peaks_max if vec[x] > 0.5]
            # if there are no candidates add day with largest peak
            if not last_day_candidates:
                try:
                    last_day_candidates.append(max(x_peaks_max, key=lambda x: vec[x]))
                except ValueError:
                    # if there are no peaks add 60
                    last_day_candidates.append(60)
            
            # choose find last day (nearest to 60) from candidates
            last_day = min(last_day_candidates, key=lambda x: abs(x - 60))
            
            # plot a line representing last day of pandemic
            ax.axvline(last_day, color='red')
            
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([0 - ymax * 0.1, ymax * 1.2])
            
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim([0 - ymax * 0.1, ymax * 1.2])
            
            fig.legend(
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 1),
                bbox_transform=ax.transAxes,
                fancybox=True,
                shadow=True)
            
            plt.tight_layout()
            cls.__show_and_save(fig=fig,
                                plot_type=f'Finding last day of pandemic up to {last_date}',
                                plot_name=(f'{voivodeship}, by {start_days_by} in {percent_of_touched_counties} '
                                           f'percent of counties'),
                                save=save,
                                show=show,
                                file_format='png')
    
    @classmethod
    def plot_pandemic_time(cls,
                           start_days_by='deaths',
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
        main_info = (f"Days considered as the period of the first phase of the pandemic.\n"
                     f"First day found based on percentage of counties with at least one "
                     f"{'infection' if start_days_by == 'infections' else 'death'} cases "
                     f"({percent_of_touched_counties}%)."
                     f" Day 0 is 04-03-2020.")
        
        ax.set_title(main_info)
        ax.set_xlabel('Voivodeship')
        ax.set_ylabel('Days of first phase of pandemic')
        
        # get start days dict
        starting_days = RealData.starting_days(
            by=start_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)
        
        ending_days = RealData.ending_days_by_death_toll_slope(
            starting_days_by=ST,
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
        
        # Plot days where first phase of pandemic was observed
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
                            plot_name=(f'Pandemic time based on {start_days_by}, '
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

        :param last_date: last possible date which can be last day of pandemic
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
        main_info = (f"Summary of designated positions in time of the first stage of the pandemic.\n"
                     f"First day found based on percentage of counties with at least one "
                     f"infection or death case. Day 0 is 04-03-2020")
        
        ax.set_title(main_info)
        ax.set_xlabel(f'Voivodeship')
        ax.set_ylabel(f'Days of first phase of pandemic')
        
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
                starting_days='deaths',
                percent_of_touched_counties=percent_of_deaths_counties,
                last_date=last_date,
                death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
                death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
            )
        
        ending_days_by_infections = \
            RealData.ending_days_by_death_toll_slope(
                starting_days='infections',
                percent_of_touched_counties=percent_of_infected_counties,
                last_date=last_date,
                death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
                death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
            )
        
        # get list of voivodeships to iterate over them while getting pandemic duration
        starting_days_by_infections = sort_dict_by_values(starting_days_by_infections)
        voivodeships_synchro = starting_days_by_infections.keys()
        
        # prepare plotting colors and transparencies
        color_deaths = 'C0'
        color_infections = 'C1'
        alpha_deaths = 0.4
        alpha_infections = 0.4
        
        # Plot days where first phase of pandemic was observed
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
        deaths_patch = mpatches.Patch(color='None',
                                      label=f'Days found by {percent_of_deaths_counties}% '
                                            f'of counties where someone died')
        infections_patch = mpatches.Patch(color='None',
                                          label=f'Days found by {percent_of_infected_counties}% '
                                                f'of counties where someone got infected')
        
        # create legend entries to a legend
        leg = ax.legend(handles=[deaths_patch, infections_patch])
        
        # change text color of legend entries description
        colors = [color_deaths, color_infections]
        for i, (h, t) in enumerate(zip(leg.legendHandles, leg.get_texts())):
            t.set_color(colors[i])
        
        # remove line/marker symbol from legend (leave only colored description)
        for i in leg._legend_handle_box.get_children()[0].get_children():  # noqa (suppres warning)
            i.get_children()[0].set_visible(False)
        
        # rotate x labels (voivodeship names)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        
        # improve y lower limit
        ax.set_ylim([0, None])
        
        plt.tight_layout()
        cls.__show_and_save(fig=fig,
                            plot_type='Days of pandemic comparison',
                            plot_name=(f'Counties death {percent_of_deaths_counties} percent,   '
                                       f'infections {percent_of_infected_counties} percent'),
                            save=save,
                            show=show)


class SimulatedVisualisation(object):
    """
    Class responsible for making visualisation of real and simulated data.

    Methods:
        * __show_and_save --> responsible for showing, saving and closing figure.

        * show_real_death_toll_for_voivodeship_shifted_by_hand --> responsible for
            plotting real death toll shifted among X axis such a given day, let's
            say day 10, looks like the beginning of pandemic in given voivodeship.
            Shift was estimated by looking since how many deaths death toll looks
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

        * max_death_toll_fig_param1_xAxis_param2_series_param3 --> plot max death toll as follow:
            - one figure for each param1 value.
            - param2 as X axis.
            - one line for each param3 value.

        * __plot_matched_real_death_toll_to_simulated --> gets y1 and y2 data arrays
            and some X length. Plots, y1, y2, and y1 shifted to best match y2 on given
            X interval. Just make plot without any axis naming etc, only make labels
            ['simulated', 'real', 'real_shifted'].

    """
    
    def __init__(self):
        pass
    
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
        :param plot_type: general description of a plot type, each unique type will have it's own folder
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
            Params are read from latest file in directory.
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
        dict_result = Results.get_single_results(not_avg_data_directory=not_avg_directory)
        
        # get beta value from latest file in avg_directory
        fnames = all_fnames_from_dir(directory=avg_directory)
        latest_file = max(fnames, key=os.path.getctime)
        variable_params = variable_params_from_fname(fname=latest_file)
        beta = variable_params[TRANSLATE.to_short('beta')]
        
        # create figure, axes, titles and labels
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title(f"Death toll stochastic for {voivodeship} with " + r"$\beta$=" + f"{beta}")
        ax.set_xlabel('t, days')
        ax.set_ylabel(r'Death toll')
        
        # plot not avg data (stochastic)
        for df in dict_result.values():
            ax.plot(df['Day'], df['Dead people'])
        
        plt.tight_layout()
        
        cls.__show_and_save(fig=fig,
                            dir_to_data=avg_directory,
                            plot_type=f"Death toll stochastic",
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
        """
        Plots averaged DataCollector model_reporter from simulations.
        One figure for one mortality-visibility pair.
        One line for each beta.


        :param directory: directory to averaged simulated data
        :type directory: str
        :param model_reporter: name of model reporter from model
            DataCollector which data will be plotted. Allowed inputs:
            ['Dead people', 'Infected toll']
        :type model_reporter: str
        :param parameter: parametr which will be swept (one plot line for each value)
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
        possible_params = TRANSLATE.to_short(['beta', 'mortality', 'visibility'])
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
        
        # for each pair create new figure
        for plot_id in range(num_of_plots):
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # prepare beata_string which prints itself as beta symbol instead of just 'beta'
            # to use it while polot instead of use f'{param}' use f'{param.replace('beta', beta_str)}'
            beta_str = r'$\beta$'
            
            # make title which includes voivodeship and normalization
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
            
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
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
                    ax.plot(df['Day'], df[model_reporter] / np.max(df[model_reporter]), label=legend,
                            color=colors[line_id])
                else:
                    ax.plot(df['Day'], df[model_reporter], label=legend, color=colors[line_id])
                
                model_reporter_max_values[line_id] = np.max(df[model_reporter])
            
            # also plot real live pandemic data if you want
            if plot_real and voivodeship is not None:
                legend = 'Real data'
                if model_reporter == 'Dead people':
                    real_data = RealData.get_real_death_toll()
                elif model_reporter == 'Infected toll':
                    real_data = RealData.get_real_infected_toll()
                else:
                    raise ValueError(f'model_reporter {model_reporter} not '
                                     f'implemented in RealData class')
                
                y = np.array(real_data.loc[voivodeship].to_list())
                
                # Find first day for which real life data is
                # grater than greatest simulated, to nicely truncate plot.
                last_real_day = np.argmax(y > max(model_reporter_max_values))
                if last_real_day == 0:
                    last_real_day = len(y)
                
                # plot initial part of real death toll
                x = range(last_real_day)
                y = y[:last_real_day]
                ax.plot(x, y, label=legend, color='black', linewidth=4, alpha=0.5)
            
            # set legend entries in same order as lines were plotted (which was by ascending beta values)
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
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
            param2 on x axis.
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
        check_uniqueness_and_correctness_of_params(param1=param1, param2=param2, param3=param3)
        
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
        grouped_fnames = group_fnames_by_param1_param2_param3(directory=directory,
                                                              param1=param1,
                                                              param2=param3,
                                                              param3=param2)
        
        [print(grouped_fnames[0][i][0]) for i in range(len(grouped_fnames[0]))]
        
        # get range for ijk (explained above)
        num_of_plots, num_of_lines, num_of_points_in_line = grouped_fnames.shape
        
        # set colormap (each line will have it's own unique color)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
        beta_str = r'$\beta$'
        
        # for each param1 value create a new figure
        for plot_id in range(num_of_plots):
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # set title title
            param1_value = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])[param1]
            
            main_title = (f"Death toll max for "
                          f"{param1.replace('beta', beta_str)}={float(param1_value) * 100:.0f}%  "
                          f"(after {get_last_simulated_day(fname=grouped_fnames[plot_id][0][0]) + 1} days)\n")
            
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
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
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)
            
            # set lower y limit to 0
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            
            # handles showing and saving simulation
            main_title = main_title.replace('\n', '').replace(r'$\beta$', 'beta')
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
                                         save=True):
        
        def plot_matched_real_death_toll_to_simulated(
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

            As it is a general function it returns plt.figure, so that it can be modified
            by function one level above.

            """
            # match simulated data to real by shifting along x axis
            shift, error = TuningModelParams._find_best_x_shift_to_match_plots(  # noqa (supress warning)
                y1_reference=simulated_death_toll,
                y2=real_death_toll,
                y2_start_idx=starting_day,
                y2_end_idx=ending_day)
            
            # get last day for which data will be plotted
            last_day_to_search = list(RealData.get_real_death_toll().columns).index(last_date_str)
            
            # truncate data up to given date
            real_death_toll = real_death_toll[:last_day_to_search]
            simulated_death_toll = simulated_death_toll[:last_day_to_search - shift]
            
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
            
            # plot vertical lines which will mark segment of real death toll
            # for which simulated death toll was tuned
            ax1.axvline(starting_day, color='red')
            ax1.axvline(ending_day, color='red', label='dni do których dopasowywano model')
            
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
            ax1.set_xticklabels([f'{x:.0f}' if x != 0 else '2020-03-04' for x in ax1.get_xticks().tolist()])
            
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
            simulated_recovered_toll = simulated_recovered_toll[:last_day_to_search - shift]
            
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
                xytext=(ending_day - 10, real_recovered_toll[ending_day] + .5 * y_annotate_offset),
                color=color_real,
                fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none", lw=2),
            )
            
            ax2.annotate(
                f"{simulated_recovered_toll[ending_day - shift] / 1000:.1f}k",
                xy=(ending_day, simulated_recovered_toll[ending_day - shift]),
                xytext=(ending_day - 10, simulated_recovered_toll[ending_day - shift] + 2 * y_annotate_offset),
                color=color_simulated,
                fontsize=14,
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none", lw=2))
            
            def format_number(*args):
                """To write numbers of recovered with 'k' suffix. 1500 = 1.5k"""
                
                value = args[0]
                if value != 0:
                    formatter = '{:1.0f}k'.format(value * 0.001)
                else:
                    formatter = '{:0.0f}'.format(value)
                return formatter
            
            ax2.yaxis.set_major_formatter(format_number)
            ax2.axvline(ending_day, color='red')
            
            # annotate day 0 as 2020-03-04
            ax2.set_xticks(ax2.get_xticks().tolist()[1:-1])
            ax2.set_xticklabels([f'{x:.0f}' if x != 0 else '2020-03-04' for x in ax2.get_xticks().tolist()])
            
            ax2.legend(loc='upper left')
            
            return ax1, ax2, fig
        
        def make_plot(sub_df):
            """
            Call 'plot_matched_real_death_toll_to_simulated' and add
            new details to plot returned by it.
            """
            
            # get real death toll
            death_tolls = RealData.get_real_death_toll()
            recovered_tolls = RealData.get_real_infected_toll()
            
            # from summary df_one_voivodeship_only get row with best tuned params
            best_run = sub_df.iloc[0]
            
            # read summary details from that row
            voivodeship = best_run['voivodeship']
            starting_day = best_run['first day']
            ending_day = best_run['last day']
            visibility = best_run['visibility']
            mortality = best_run['mortality']
            beta = best_run['beta']
            fname = best_run['fname']
            
            # read avg_df which contains course of pandemic in simulation
            avg_df = pd.read_csv(fname)
            
            # make plot (only with data labels; no axis, titles etc.)
            ax1, ax2, fig = plot_matched_real_death_toll_to_simulated(
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
            
            ax1.set_title('Zgony')
            ax1.set_xlabel('t, dni')
            ax1.set_ylabel('Suma zgonów')
            
            ax2.set_title('Wyzdrowienia')
            ax2.set_xlabel('t, dni')
            ax2.set_ylabel('Suma wyzdrowień')
            
            # Hide the right and top spines
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            
            plt.tight_layout()
            
            # SAVING ------------------------------------------------------------------------------
            # handles showing and saving simulation
            main_title = f"{voivodeship} finest model params tuning"
            plot_type = 'Best death toll fits, beta and mortality'
            plot_name = " ".join(main_title.split())
            
            # save in general plots folder
            if save:
                save_dir = Directories.ABM_DIR + '/RESULTS/plots/' + plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir + plot_name + '.png')
            
            # show and save in avg_simulation folder
            cls.__show_and_save(fig=fig,
                                dir_to_data=os.path.split(fname)[0] + '/',
                                plot_type='Auto fit by percent_of_touched_counties',
                                plot_name=" ".join(main_title.split()),
                                save=save,
                                show=show)
        
        # get summary df
        df_tuning = TuningModelParams.get_tuning_results()
        
        # make plot gor each voivodeship which was tuned
        for voivodeship_tuned in df_tuning['voivodeship'].unique():
            make_plot(sub_df=df_tuning.loc[df_tuning['voivodeship'] == voivodeship_tuned])
    
    @classmethod
    def plot_best_beta_mortality_pairs(cls,
                                       pairs_per_voivodeship: int,
                                       show=True,
                                       save=False):
        
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        fig.suptitle(f"Najlepiej dopasowane wartości "r'$\beta$'f" i śmiertelności"
                     f" dla poszczególnych województw.")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Śmiertelność")
        
        best_df = TuningModelParams.get_n_best_tuned_results(
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
        
        good_line = mlines.Line2D([], [], color='grey', marker='o', mec='k',
                                  markersize=20, ls='None', label='Dobre dopasowanie')
        
        bad_line = mlines.Line2D([], [], color='grey', marker='X', mec='k',
                                 markersize=20, ls='None', label='Złe dopasowanie')
        
        ax.legend(handles=[good_line, bad_line],
                  labelspacing=2,
                  borderpad=1.2,
                  loc='lower right')
        
        texts = []
        # annotate voivodeships
        for x_pos, y_pos, voivodeship in zip(x, y, voivodeships):
            texts.append(ax.text(
                x_pos,
                y_pos,
                voivodeship,
                ha='center',
                va='center',
                bbox=dict(boxstyle="square,pad=0.3",
                          fc=mColors.to_rgba('lightgrey')[:-1] + (1,),
                          ec="none"),
            ))
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
    def all_death_toll_from_dir_by_fixed_params(cls, fixed_params: dict, c_norm_type='log'):
        """Plot simple death toll dynamic to ensure that params in model works fine.

        fixed_params: dict defining folder in ABM/RESULTS which will be plotted.

        Note: fnames should contain exactly 4 params, where 3 of them should be
            beta, mortality, visibility. Example fname:
            'Id=0001__p_inf_cust=2.34__b=0.025__m=0.02__v=0.65.csv'
        """
        
        def get_param_name_from_fname(fname):
            """Returns first param name found in fname except from: 'beta', 'mortality', 'visibility'."""
            
            fname = os.path.split(fname)[1]  # get fname without prev dirs
            fname_splitted = fname_to_list(fname)  # convert fname to list like ['Beta=0.036', ...]
            
            for param in fname_splitted:
                if not ('beta' in param.lower()
                        or 'mortality' in param.lower()
                        or 'visibility' in param.lower()):
                    return param.split('=')[0].lower()
        
        def sort_helper(fname, parameter_name):
            fname = os.path.split(fname)[1]  # get fname without prev dirs
            fname_splitted = fname_to_list(fname)  # convert fname to list like ['Beta=0.036', ...]
            
            # make dict {'beta': 0.036, ...}
            fname_dict = {item.split('=')[0].lower(): float(item.split('=')[1]) for item in fname_splitted}
            return fname_dict[parameter_name.lower()]
        
        # Get folder name (which contains csv files)
        folder = find_folder_by_fixed_params(
            directory=Directories.AVG_SAVE_DIR,
            params=fixed_params)
        folder += 'raw data/'
        
        # Get fnames and sort them by param values
        fnames = all_fnames_from_dir(directory=folder)
        param_name = get_param_name_from_fname(fnames[0])
        fnames = sorted(fnames, key=lambda fname: sort_helper(fname, param_name))
        param_name = TRANSLATE.to_short(param_name)
        
        # Get variable params from fnames and dfs from its csv files
        variable_params = [TRANSLATE.to_short(variable_params_from_fname(fname)) for fname in fnames]
        dfs = [pd.read_csv(fname) for fname in fnames]
        
        # Prepare  fig and ax
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        sns.despine(fig=fig, ax=ax)
        
        fig.suptitle("Suma zgonów")
        ax.set_title(f"Różne wartości '{param_name.replace('_', ' ')}'")
        
        # Create rainbow colormap and colors; based on param values
        # get min and max param values
        param_values = np.array([float(params[param_name]) for params in variable_params])
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
                log_norm = mColors.LogNorm(vmin=min_param, vmax=max_param + min_param)
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


def main():
    # VisRealDT(last_day_to_plot=200).plot()
    
    VisRealDTShiftedByHand().plot()


if __name__ == '__main__':
    main()
