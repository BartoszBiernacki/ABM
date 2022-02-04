from disease_spread_model.data_processing.visualisation import *
from disease_spread_model.config import Config


# [3] ---------------------------------------------------------------------------------------------------------------
shift, error = find_best_x_shift_to_match_plots(
    y1_reference=[i * i for i in range(11)],
    y2=[50, 30, 20] + [i * i for i in range(11)],
    y2_start=3,
    y2_end=8)

print(f"Shift = {shift} with fit error = {error}")
# -------------------------------------------------------------------------------------------------------------------


# [16]  ******************************************************
RealVisualisation.plot_pandemic_starting_days_by_touched_counties(
    percent_of_touched_counties=40,
    normalize_by_population=True,
    save=True,
    show=True)
#  ***********************************************************

# [17]  ******************************************************
RealVisualisation.show_real_death_toll(voivodeships=['all'],
                                       last_day=100,
                                       normalized=False,
                                       show=True,
                                       save=True)
#  ***********************************************************

# [18]  ******************************************************
SimulatedVisualisation.show_real_death_toll_for_voivodeship_shifted_by_hand(
    directory_to_data=Config.avg_directory,
    voivodeship=Config.voivodeship,
    shift_simulated=True,
    save=True)
#  ***********************************************************

# [19]  ******************************************************
SimulatedVisualisation.plot_stochastic_1D_death_toll_dynamic(
    avg_directory=Config.avg_directory,
    not_avg_directory=Config.not_avg_directory,
    voivodeship=Config.voivodeship,
    show=True,
    save=True)
#  ***********************************************************

# [20]  ******************************************************
RealVisualisation.show_real_death_toll_shifted_by_hand(
    starting_day=10,
    day_in_which_colors_are_set=60,
    last_day=100,
    directory_to_data=None,
    shift_simulated=True,
    show=True,
    save=False)
#  ***********************************************************

# [21]  ******************************************************
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

# [22]  ******************************************************
SimulatedVisualisation.plot_auto_fit_death_toll(
    directory=Config.avg_directory,
    voivodeship=Config.voivodeship,
    percent_of_touched_counties=Config.percent_of_touched_counties,
    days_to_fit=Config.days_to_fit_death_toll,
    ignore_healthy_counties=True,
    show=True,
    save=True)
#  ***********************************************************

# [23]  ******************************************************
SimulatedVisualisation.max_death_toll_fig_param1_xAxis_param2_series_param3(
    directory=Config.avg_directory,
    param1='visibility',
    param2='beta',
    param3='mortality',
    show=True,
    save=False
)
#  ***********************************************************

# [24]  ******************************************************


#  ***********************************************************

# [16]  ******************************************************

#  ***********************************************************

# [16]  ******************************************************

#  ***********************************************************

# [16]  ******************************************************

#  ***********************************************************

# [16]  ******************************************************

#  ***********************************************************
