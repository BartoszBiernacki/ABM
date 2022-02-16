"""Making all kind of plots to visualize real, simulated or both data."""
import datetime

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.data_processing.avg_results import Results
from disease_spread_model.data_processing.real_data import RealData
from disease_spread_model.model.model_adjustment import TuningModelParams

from disease_spread_model.config import Config
from disease_spread_model.model.my_math_utils import *


class RealVisualisation(object):
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
    
    def __init__(self):
        pass
    
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
            save_dir = Config.ABM_dir + '/RESULTS/plots/' + plot_type + '/'
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
        
        # set title and axis labels
        norm_info = ''
        if normalized:
            norm_info = ' (normalized)'
        
        ax.set_title(f"Death toll{norm_info}. Data for {last_day} days since first case i.e. 04-03-2020.\n"
                     f"The colors of the lines are matched with regard to the urbanization factor "
                     f"(red for the biggest value).")
        if normalized:
            ax.set_ylabel(f"Death toll / Day toll(day={last_day})")
        else:
            ax.set_ylabel(f"Death toll")
        ax.set_xlabel("t, day since first day of collecting data i.e. 04-03-2020")
        
        # set colormap
        cmap = plt.get_cmap('rainbow')
        # set proper number of colors equal to number of lines
        if 'all' in voivodeships:
            num_of_lines = len(RealData.get_voivodeships())
            colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
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
            if normalized:
                if np.max(y[: last_day]) > 0:
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
        
        cls.__show_and_save(fig=fig,
                            plot_type=f'Real death toll, since 04-03-2020',
                            plot_name=cls._get_plot_name_from_title(ax.get_title()),
                            save=save,
                            show=show)
    
    @classmethod
    def show_real_death_toll_shifted_by_hand(cls,
                                             starting_day=10,
                                             day_in_which_colors_are_set=60,
                                             last_day=100,
                                             directory_to_data=None,
                                             shift_simulated=False,
                                             save=False,
                                             show=True):
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
                    beta = float(variable_params_from_fname(fname=fname)[r'$\beta$'])
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
        main_info = (f"Day that is consider as starting day of pandemic "
                     f"for given voivodeship and death toll in that day.\n"
                     f" Day 0 is 04-03-2020.")
        
        ax.set_title(main_info)
        ax.set_xlabel(f'Voivodeship')
        ax.set_ylabel(f'Day number since 04-03-2020 which is considered to be the beginning of a pandemic')
        
        # get starting day of pandemic by percent of touched counties
        starting_days_deaths = \
            RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                percent_of_touched_counties=percent_of_death_counties,
                ignore_healthy_counties=False)
        
        starting_days_infections = \
            RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                percent_of_touched_counties=percent_of_infected_counties)
        
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
                                      ):
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
        if start_days_by == 'deaths':
            start_days = RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                percent_of_touched_counties=percent_of_touched_counties,
                ignore_healthy_counties=True)
        elif start_days_by == 'infections':
            start_days = RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                percent_of_touched_counties=percent_of_touched_counties)
        else:
            raise ValueError(f'start_days_by has to be "deaths" or "infections", but {start_days_by} was given')

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
            last_day = min(last_day_candidates, key=lambda x: abs(x-60))
            
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
        death or infected counties. Last day got from slope of feath toll.
        """
    
        # make fig and ax
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
    
        # make ax title and axis labels
        main_info = (f"Days considered as the period of the first phase of the pandemic.\n"
                     f"First day found based on percentage of counties with at least one "
                     f"{'infection' if start_days_by=='infections' else 'death'} cases "
                     f"({percent_of_touched_counties}%)."
                     f" Day 0 is 04-03-2020.")
    
        ax.set_title(main_info)
        ax.set_xlabel(f'Voivodeship')
        ax.set_ylabel(f'Days of first phase of pandemic')
    
        # get starting day of pandemic by percent of touched counties
        if start_days_by == 'deaths':
            starting_days = \
                RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                    percent_of_touched_counties=percent_of_touched_counties,
                    ignore_healthy_counties=False)
        else:
            starting_days = \
                RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                    percent_of_touched_counties=percent_of_touched_counties)
            
        ending_days = RealData.ending_days_by_death_toll_slope(
            start_days_by=start_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date=last_date,
            death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
            death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
        )
    
        # Get dict {voivodeship: pandemic_duration_in_days}
        pandemic_duration = {}
        for voivodeship in starting_days.keys():
            pandemic_duration[voivodeship] = ending_days[voivodeship] - starting_days[voivodeship]
        
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
        starting_days_by_deaths = \
            RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                percent_of_touched_counties=percent_of_deaths_counties,
                ignore_healthy_counties=False)
    
        # get starting day of pandemic by infected counties
        starting_days_by_infections = \
            RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                percent_of_touched_counties=percent_of_infected_counties)

        ending_days_by_deaths = \
            RealData.ending_days_by_death_toll_slope(
                start_days_by='deaths',
                percent_of_touched_counties=percent_of_deaths_counties,
                last_date=last_date,
                death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
                death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
            )
        
        ending_days_by_infections = \
            RealData.ending_days_by_death_toll_slope(
                start_days_by='infections',
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
        for i in leg._legend_handle_box.get_children()[0].get_children():   # noqa (suppres warning)
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
    def __show_and_save(cls, fig, dir_to_data, plot_type, plot_name, save, show):
        """Function that shows and saves figures

        :param fig: figure to be shown/saved
        :type fig: matplotlib figure
        :param dir_to_data: directory to simulated data, used in image output directory
        :type dir_to_data: str
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
            else:
                save_dir = plot_type + '/'
            
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + plot_name + '.pdf')
        
        if show:
            plt.show()
        
        plt.close(fig)
    
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
        beta = variable_params['$\\beta$']
        
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
        :param parameter: parametr which will be swept (one line for each value)
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
        possible_params = ['beta', 'mortality', 'visibility']
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
    def __plot_matched_real_death_toll_to_simulated(cls,
                                                    y1_simulated,
                                                    y2_real,
                                                    y2_start=None,
                                                    y2_end=None,
                                                    show=True):
        """
        Assumes y(x=0) = y[0], y(x=1) = y[1] and so on.
        Function moves y2 data along X axis to find out for what shift subset of y2 = y2[start: end]
        best matches subset of the same length of y1. Then plot y1, y2, y2_shifted.

        As it is a general function it returns plt.figure, so that it can be modified
        by function one level above.

        """
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # prepare line colors
        color_real = 'lime'
        color_real_shifted = 'forestgreen'
        color_simulated = 'gold'
        
        # plot simulated and real data
        ax.plot(range(len(y1_simulated)), y1_simulated, label='simulated', color=color_simulated)
        ax.plot(range(len(y2_real)), y2_real, label='real', color=color_real)
        
        # if y boundaries are given plot real shifted and matching segments
        if y2_start and y2_end:
            # plot part of real data to which the simulated data will be matched
            ax.plot(np.arange(y2_start, y2_end), y2_real[y2_start: y2_end],
                    color=color_real, linewidth=5)
            
            # match simulated data to real by shifting along x axis
            shift, error = TuningModelParams._find_best_x_shift_to_match_plots(     # noqa (supress warning)
                y1_reference=y1_simulated,
                y2=y2_real,
                y2_start_idx=y2_start,
                y2_end_idx=y2_end)
            
            if shift < 0:
                y2_new = [np.NaN] * (-shift) + y2_real.to_list()
            else:
                y2_new = y2_real[shift:]
            
            # plot shifted real data
            ax.plot(range(len(y2_new)), y2_new, label='real shifted', color=color_real_shifted)
            # plot matching segment of shifted real data
            ax.plot(np.arange(y2_start - shift, y2_end - shift),
                    y2_new[y2_start - shift: y2_end - shift],
                    color=color_real_shifted,
                    linewidth=5)
            
            # plot segment of simulated data for which data were matched
            ax.plot(np.arange(y2_start - shift, y2_end - shift),
                    y1_simulated[y2_start - shift: y2_end - shift],
                    color=color_simulated,
                    linewidth=5)
            
            ax.set_xlim(0, 265)
            ax.set_ylim(0, 5 * y2_real[y2_end])
        else:
            ax.set_xlim(0, y2_end)
            ax.set_ylim(0, 1.1 * y2_real[y2_end])
        
        ax.legend()
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax
    
    @classmethod
    def plot_auto_fit_death_toll(cls,
                                 directory: str,
                                 voivodeship: str,
                                 percent_of_touched_counties: int,
                                 start_day_by='deaths',
                                 death_toll_smooth_out_win_size=21,
                                 death_toll_smooth_out_polyorder=3,
                                 last_date='2020-07-01',
                                 show=True,
                                 save=False):
        """
        Assumes y(x=0) = y[0], y(x=1) = y[1] and so on.
        Function moves y2 data along X axis to find out for what shift, subset of real death toll
        best matches subset of the same length of simulated death toll.
        Then plot: real death toll, simulated death toll, real death toll shifted.

        Simulated data is the latest file in directory.
        Start results from the percent_of_touched_counties.
        End = start + days_to_fit
        """
        real_death_toll = RealData.get_real_death_toll()
        
        fnames = all_fnames_from_dir(directory=directory)
        latest_file = max(fnames, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        if start_day_by == 'deaths':
            starting_day = RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                percent_of_touched_counties=percent_of_touched_counties,
                ignore_healthy_counties=False)
        elif start_day_by == 'infections':
            starting_day = RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                percent_of_touched_counties=percent_of_touched_counties)
        else:
            raise ValueError(f"start_day_based_on must equal to 'deaths' or 'infections', but "
                             f"{start_day_by} was given!")

        ending_day = RealData.ending_days_by_death_toll_slope(
            start_days_by=start_day_by,
            percent_of_touched_counties=percent_of_touched_counties,
            last_date=last_date,
            death_toll_smooth_out_win_size=death_toll_smooth_out_win_size,
            death_toll_smooth_out_polyorder=death_toll_smooth_out_polyorder,
        )
        
        fig, ax = cls.__plot_matched_real_death_toll_to_simulated(
            y1_simulated=df['Dead people'],
            y2_real=real_death_toll.loc[voivodeship],
            y2_start=starting_day[voivodeship],
            y2_end=starting_day[voivodeship] + ending_day[voivodeship],
            show=False)
        
        ax.set_title(f'Reported death toll for {voivodeship} shifted to best match simulated data on '
                     f'the selected days range days.\n'
                     f'Selected day range starts in a day when in {percent_of_touched_counties}% of '
                     f'counties first {"death" if start_day_by == "deaths" else "infection"} '
                     f'was reported.')
        ax.set_xlabel('t, days')
        ax.set_ylabel('death toll')
        plt.tight_layout()
        
        # handles showing and saving simulation
        main_title = (f"start day based on {start_day_by},   "
                      f"{percent_of_touched_counties} percent of counties")
        cls.__show_and_save(fig=fig,
                            dir_to_data=directory,
                            plot_type='Auto fit by percent_of_touched_counties',
                            plot_name=" ".join(main_title.split()),
                            save=save,
                            show=show)
        
    @classmethod
    def plot_death_toll_for_best_tuned_model_params(cls):
        # TODO get extra list param like [beta, mortality] and add those to legend
        # TODO plot also infected toll
        
        def _make_plot(sub_df):
            real_death_toll = RealData.get_real_death_toll()
            best_run = sub_df.iloc[0]

            voivodeship = best_run['voivodeship']
            starting_day = best_run['first day']
            ending_day = best_run['last day']
            fname = best_run['fname']
            avg_df = pd.read_csv(fname)

            fig, ax = cls.__plot_matched_real_death_toll_to_simulated(
                y1_simulated=avg_df['Dead people'],
                y2_real=real_death_toll.loc[voivodeship],
                y2_start=starting_day,
                y2_end=ending_day,
                show=False)

            ax.set_title(f'Reported death toll for {voivodeship} shifted to best match best simulated data on '
                         f'the selected days range.')
            ax.set_xlabel('t, days')
            ax.set_ylabel('death toll')
            plt.tight_layout()

            # handles showing and saving simulation
            main_title = f"finest model params tuning"
            cls.__show_and_save(fig=fig,
                                dir_to_data=os.path.split(fname)[0] + '/',
                                plot_type='Auto fit by percent_of_touched_counties',
                                plot_name=" ".join(main_title.split()),
                                save=True,
                                show=True)
 
        df_tuning = TuningModelParams.get_tuning_results()
        for voivodeship_tuned in df_tuning['voivodeship'].unique():
            _make_plot(sub_df=df_tuning.loc[df_tuning['voivodeship'] == voivodeship_tuned])
            
    @classmethod
    def plot_best_beta_mortality_pairs(cls, pairs_per_voivodeship: int):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        ax.set_title(f"Best tuned "r'$\beta$'f" and mortality for each voivodeship")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("mortality")
        
        best_df = TuningModelParams.get_n_best_tuned_results(
            n=pairs_per_voivodeship)
        
        color_list = ['green', 'red']
        
        x = best_df['beta']
        y = best_df['mortality']
        c = []
        
        for i in range(len(x)):
            if best_df['fit error per day'].iloc[i] < 100:
                c.append(color_list[0])
            else:
                c.append(color_list[1])

        good_line = mlines.Line2D([], [], color=color_list[0], marker='o', mec='k',
                                  markersize=10, ls='None', label='Fits well')

        bad_line = mlines.Line2D([], [], color=color_list[1], marker='o', mec='k',
                                 markersize=10, ls='None', label='Fits bad')
        
        ax.scatter(x=x, y=y, c=c, s=100, edgecolors='black')

        ax.legend(handles=[good_line, bad_line])
        plt.show()


if __name__ == '__main__':
    
    # TODO design scatter like plot with best beta-mortality pairs and fit error

    import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()

    SimulatedVisualisation.plot_death_toll_for_best_tuned_model_params()

    # SimulatedVisualisation.plot_best_beta_mortality_pairs(
    #     pairs_per_voivodeship=1)
    
    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)
    pass
