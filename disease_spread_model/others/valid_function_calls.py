from disease_spread_model.model.model_adjustment import TuningModelParams
from disease_spread_model.data_processing.visualisation import *
from disease_spread_model.config import Config


# [1] ---------------------------------------------------------------------------------------------------------------
shift, error = TuningModelParams._find_best_x_shift_to_match_plots(
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
FindLastDayAnim.make_animations(voivodeships=['opolskie'],
                                start_days_by='infections',
                                fps=50,
                                show=True,
                                save=False)
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
    first_days =\
        RealData.get_starting_days_for_voivodeships_based_on_district_infections(
            percent_of_touched_counties=80
        )
    last_days = RealData.get_ending_days_for_voivodeships_based_on_death_toll_derivative(
        starting_days=first_days
    )
    
    TuningModelParams.find_beta_for_voivodeships(
        voivodeships=['łódzkie'],
        starting_days=first_days,
        ending_days=last_days,
        mortality=2,
        visibility=0.65)
    
    TuningModelParams.find_mortality_for_voivodeships(
        voivodeships=['łódzkie'],
        starting_days=first_days,
        ending_days=last_days,
        beta=0.025,
        visibility=0.65)

    TuningModelParams.super_optimizing(
        voivodeships=['opolskie'],
        starting_days=first_days,
        ending_days=last_days,
        visibility=0.65,
        beta_init=0.025,
        mortality_init=2,
        simulation_runs=12,
    )

    df = TuningModelParams.get_tuning_results()
    [print(item['fname']) for _, item in df.iterrows()]
#  ***********************************************************
