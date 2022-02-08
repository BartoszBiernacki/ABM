from disease_spread_model.model.model_adjustment import TuningModelParams
from disease_spread_model.data_processing.visualisation import *
from disease_spread_model.config import Config


# [1] ---------------------------------------------------------------------------------------------------------------
shift, error = TuningModelParams.find_best_x_shift_to_match_plots(
    y1_reference=[i * i for i in range(11)],
    y2=[50, 30, 20] + [i * i for i in range(11)],
    y2_start_idx=3,
    y2_end_idx=8)

print(f"Shift = {shift} with fit error = {error}")
# -------------------------------------------------------------------------------------------------------------------


# [2]  ******************************************************
RealVisualisation.plot_pandemic_starting_days_by_touched_counties(
    percent_of_death_counties=20,
    percent_of_infected_counties=80,
    normalize_by_population=True,
    save=True,
    show=True)
#  ***********************************************************

# [3]  ******************************************************
RealVisualisation.show_real_death_toll(voivodeships=['all'],
                                       last_day=100,
                                       normalized=False,
                                       show=True,
                                       save=True)
#  ***********************************************************

# [4]  ******************************************************
SimulatedVisualisation.show_real_death_toll_for_voivodeship_shifted_by_hand(
    directory_to_data=Config.avg_directory,
    voivodeship=Config.voivodeship,
    shift_simulated=True,
    save=True)
#  ***********************************************************

# [5]  ******************************************************
SimulatedVisualisation.plot_stochastic_1D_death_toll_dynamic(
    avg_directory=Config.avg_directory,
    not_avg_directory=Config.not_avg_directory,
    voivodeship=Config.voivodeship,
    show=True,
    save=True)
#  ***********************************************************

# [6]  ******************************************************
RealVisualisation.show_real_death_toll_shifted_by_hand(
    starting_day=10,
    day_in_which_colors_are_set=60,
    last_day=100,
    directory_to_data=None,
    shift_simulated=True,
    show=True,
    save=False)
#  ***********************************************************

# [7]  ******************************************************
SimulatedVisualisation.plot_1D_modelReporter_dynamic_parameter_sweep(
    directory=Config.avg_directory,
    model_reporter='Dead people',
    parameter='beta',
    normalized=False,
    plot_real=True,
    show=True,
    save=False
)

#  ***********************************************************

# [8]  ******************************************************
SimulatedVisualisation.plot_auto_fit_death_toll(
        directory=Config.avg_directory,
        voivodeship=Config.voivodeship,
        percent_of_touched_counties=20,
        start_day_based_on='deaths',
        show=True,
        save=False
    )
#  ***********************************************************

# [9]  ******************************************************
SimulatedVisualisation.max_death_toll_fig_param1_xAxis_param2_series_param3(
    directory=Config.avg_directory,
    param1='visibility',
    param2='beta',
    param3='mortality',
    show=True,
    save=False
)
#  ***********************************************************

# [10]  ******************************************************
FindLastDayAnim.save_animations(voivodeships=['all'], fps=50)
FindLastDayAnim.show_animations(voivodeships=['all'], fps=50)
#  ***********************************************************

# [11]  ******************************************************
RealVisualisation.plot_pandemic_starting_days_by_touched_counties(
    percent_of_death_counties=20,
    percent_of_infected_counties=80,
    normalize_by_population=False,
    save=True,
    show=True
)
#  ***********************************************************

# [12]  ******************************************************
for criterion in ['infections', 'deaths']:
    RealVisualisation.plot_pandemic_time(
        based_on=criterion,
        percent_of_touched_counties=20,
        show=True,
        save=True)
#  ***********************************************************

# [13]  ******************************************************
    RealVisualisation.compare_pandemic_time_by_infections_and_deaths(
        percent_of_deaths_counties=20,
        percent_of_infected_counties=80,
        save=True,
        show=True)
#  ***********************************************************

# [16]  ******************************************************

#  ***********************************************************
