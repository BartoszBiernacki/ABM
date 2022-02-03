from data_visualisation import *
from config import Config


# [1] ---------------------------------------------------------------------------------------------------------------
from my_models.mesa_grid_star.avg_results import get_single_results
from my_models.mesa_grid_star.my_model_data_analysis import find_beta_for_all_voivodeships, \
    find_best_beta_for_given_mortality_visibility
from my_models.mesa_grid_star.test_visuals import RealVisualisation, RealAndSimulatedVisualisation, \
    SimulatedVisualisation

show_real_disease_death_toll_normalized(voivodeships=RealData.voivodeships,
                                        last_day=200,
                                        show=True,
                                        save=True)
# -------------------------------------------------------------------------------------------------------------------

# [2] ---------------------------------------------------------------------------------------------------------------
for percent_of_touched_counties in [20, 40, 60, 80, 100]:
    plot_pandemic_starting_days(percent_of_touched_counties=percent_of_touched_counties,
                                normalize_by_population=True,
                                save=True,
                                show=True)
# -------------------------------------------------------------------------------------------------------------------

# [3] ---------------------------------------------------------------------------------------------------------------
shift, error = find_best_x_shift_to_match_plots(y1_reference=[i * i for i in range(11)],
                                                y2=[50, 30, 20] + [i * i for i in range(11)],
                                                y2_start=3,
                                                y2_end=8)
print(f"Shift = {shift} with fit error = {error}")
# -------------------------------------------------------------------------------------------------------------------

# [4] ---------------------------------------------------------------------------------------------------------------
for minimum_deaths in [1, 2, 5, 10, 20, 50, 100, 200]:
    show_real_death_toll_shifted_to_match_death_toll_in_given_day(starting_day=10,
                                                                  day_in_which_colors_are_set=60,
                                                                  last_day=100,
                                                                  minimum_deaths=minimum_deaths,
                                                                  directory_to_data=None,
                                                                  shift_simulated=False,
                                                                  save=True,
                                                                  show=False)
# -------------------------------------------------------------------------------------------------------------------

# [5] ---------------------------------------------------------------------------------------------------------------
plot_matched_real_death_toll_to_simulated(y1_simulated=[np.sqrt(i) for i in range(250)],
                                          y2_real=[50, 30, 20] * 10 + [np.sqrt(i) for i in range(250)],
                                          y2_start=45,
                                          y2_end=200)
# -------------------------------------------------------------------------------------------------------------------

# [6] ---------------------------------------------------------------------------------------------------------------
show_real_death_tol_shifted_by_hand(starting_day=10,
                                    day_in_which_colors_are_set=60,
                                    last_day=100,
                                    directory_to_data=None,
                                    shift_simulated=True,
                                    save=True,
                                    show=False)
# -------------------------------------------------------------------------------------------------------------------

# [7] ---------------------------------------------------------------------------------------------------------------
show_real_death_toll_voivodeship_shifted_by_hand(directory_to_data=None,
                                                 voivodeship=Config.voivodeship,
                                                 starting_day=10,
                                                 day_in_which_colors_are_set=60,
                                                 last_day=100,
                                                 shift_simulated=True,
                                                 save=False,
                                                 show=True)
# -------------------------------------------------------------------------------------------------------------------

# [8] ---------------------------------------------------------------------------------------------------------------
dict_results = get_single_results(directory_to_not_averaged_results='TMP_SAVE/')

plot_stochastic_1D_death_toll_dynamic(dict_result=dict_results,
                                      avg_directory=None,
                                      voivodeship=Config.voivodeship,
                                      save=True,
                                      show=True)
# -------------------------------------------------------------------------------------------------------------------

# [9] ---------------------------------------------------------------------------------------------------------------
plot_1D_death_toll_dynamic_real_and_beta_sweep(directory=None,
                                               real_death_toll=None,
                                               voivodeship=None,
                                               normalized=False,
                                               show=True,
                                               save=True)
# -------------------------------------------------------------------------------------------------------------------

# [10] ---------------------------------------------------------------------------------------------------------------
plot_auto_fit_result(directory=None,
                     voivodeship=Config.voivodeship,
                     percent_of_touched_counties=Config.percent_of_touched_counties,
                     days_to_fit=Config.days_to_fit_death_toll,
                     ignore_healthy_counties=True)
# -------------------------------------------------------------------------------------------------------------------

# [11] ---------------------------------------------------------------------------------------------------------------
find_beta_for_all_voivodeships(mortality=2,
                               visibility=0.65)
# -------------------------------------------------------------------------------------------------------------------

# [12] ---------------------------------------------------------------------------------------------------------------
find_best_beta_for_given_mortality_visibility(voivodeship=Config.voivodeship,
                                              mortality=2,
                                              visibility=0.65,
                                              percent_of_touched_counties=percent_of_touched_counties,
                                              days_to_fit_death_toll=50,
                                              fit_iterations=8,
                                              beta_init=0.025
                                              )
# -------------------------------------------------------------------------------------------------------------------

# [13] ---------------------------------------------------------------------------------------------------------------
plot_max_death_toll_prediction_x_visibility_series_betas(directory=None,
                                                         show=True,
                                                         save=False)
# -------------------------------------------------------------------------------------------------------------------

# [14] ---------------------------------------------------------------------------------------------------------------
plot_auto_fit_result(directory=None,
                     voivodeship=Config.voivodeship,
                     percent_of_touched_counties=Config.percent_of_touched_counties,
                     days_to_fit=Config.days_to_fit_death_toll,
                     ignore_healthy_counties=True,
                     show=True,
                     save=True)
# -------------------------------------------------------------------------------------------------------------------

# [15] ---------------------------------------------------------------------------------------------------------------
plot_1D_recovered_dynamic_real_and_beta_sweep(directory=None,
                                              real_infected_toll=None,
                                              voivodeship=Config.voivodeship,
                                              normalized=False,
                                              show=True,
                                              save=True)
# -------------------------------------------------------------------------------------------------------------------

# [16]  ******************************************************
RealVisualisation.plot_pandemic_starting_days_by_touched_counties(percent_of_touched_counties=40,
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
RealAndSimulatedVisualisation. \
        show_real_death_toll_for_voivodeship_shifted_by_hand(directory_to_data=Config.avg_directory,
                                                             voivodeship=Config.voivodeship,
                                                             shift_simulated=True,
                                                             save=True)
#  ***********************************************************

# [19]  ******************************************************
SimulatedVisualisation.plot_stochastic_1D_death_toll_dynamic(avg_directory=Config.avg_directory,
                                                             not_avg_directory=Config.not_avg_directory,
                                                             voivodeship=Config.voivodeship,
                                                             show=True,
                                                             save=True)
#  ***********************************************************

# [20]  ******************************************************
RealVisualisation.show_real_death_toll_shifted_by_hand(starting_day=10,
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
