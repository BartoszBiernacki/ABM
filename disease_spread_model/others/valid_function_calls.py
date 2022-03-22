from disease_spread_model.data_processing.visualisation import *
from disease_spread_model.config import RealDataOptions

# [1]  ******************************************************
RealVisualisation.show_real_death_toll(voivodeships=['all'],
                                       last_day=100,
                                       normalized=False,
                                       show=True,
                                       save=True)
#  ***********************************************************

# [2]  ******************************************************
RealVisualisation.show_real_death_toll_shifted_by_hand(
    starting_day=10,
    day_in_which_colors_are_set=60,
    last_day=100,
    directory_to_data=None,
    shift_simulated=True,
    show=True,
    save=False)
#  ***********************************************************

# [3]  ******************************************************
RealVisualisation.plot_pandemic_starting_days_by_touched_counties(
    percent_of_death_counties=20,
    percent_of_infected_counties=80,
    normalize_by_population=True,
    save=True,
    show=True)
#  ***********************************************************

# [4]  ******************************************************
RealVisualisation.plot_last_day_finding_process(
    voivodeships=['all'],
    start_days_by='infections',
    percent_of_touched_counties=80,
    last_date='2020-07-01',
    death_toll_smooth_out_win_size=21,
    death_toll_smooth_out_polyorder=3,
    derivative_half_win_size=3,
    plot_redundant=False,
    show=True,
    save=True
)
#  ***********************************************************

# [5]  ******************************************************
RealVisualisation.plot_pandemic_time(
    start_days_by='infections',
    percent_of_touched_counties=80,
    death_toll_smooth_out_win_size=21,
    death_toll_smooth_out_polyorder=3,
    last_date='2020-07-01',
    show=True,
    save=True)
#  ***********************************************************

# [6]  ******************************************************
RealVisualisation.compare_pandemic_time_by_infections_and_deaths(
    percent_of_deaths_counties=20,
    percent_of_infected_counties=80,
    death_toll_smooth_out_win_size=21,
    death_toll_smooth_out_polyorder=3,
    last_date='2020-07-01',
    save=False,
    show=True)
#  ***********************************************************

# [7]  ******************************************************
SimulatedVisualisation.plot_stochastic_1D_death_toll_dynamic(
    avg_directory=Config.avg_directory,
    not_avg_directory=Config.not_avg_directory,
    voivodeship=Config.voivodeship,
    show=True,
    save=True)
#  ***********************************************************

# [8]  ******************************************************
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
SimulatedVisualisation.death_toll_for_best_tuned_params(
        last_date='2020-07-01',
        show=True,
        save=False
    )
#  ***********************************************************

# [11]  ******************************************************
SimulatedVisualisation.plot_best_beta_mortality_pairs(
        pairs_per_voivodeship=1,
        show=True,
        save=False)
#  ***********************************************************

# [11]  ******************************************************
SimulatedVisualisation.all_death_toll_from_dir_by_fixed_params(
    fixed_params=FolderParams.HOUSEMATE_INFECTION_PROBABILITY,
    c_norm_type='log'
)
#  ***********************************************************

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# [1 - 1]  ******************************************************
first_days = RealData.starting_days(
    by='infections',
    percent_of_touched_counties=Config.percent_of_infected_counties)

last_days = RealData.ending_days_by_death_toll_slope(
        starting_days=RealDataOptions.STARTING_DAYS_BY,
        percent_of_touched_counties=RealDataOptions.PERCENT_OF_INFECTED_COUNTIES,
        last_date=RealDataOptions.LAST_DATE,
        death_toll_smooth_out_win_size=RealDataOptions.DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_WIN_SIZE,
        death_toll_smooth_out_polyorder=RealDataOptions.DEATH_TOLL_DERIVATIVE_SMOOTH_OUT_SAVGOL_POLYORDER,
    )


TuningModelParams.optimize_beta_and_mortality(
    voivodeships=['all'],
    ignored_voivodeships=RealData.bad_voivodeships(),
    starting_days='infections',
    percent_of_touched_counties=80,
    last_date='2020-07-01',
    death_toll_smooth_out_win_size=21,
    death_toll_smooth_out_polyorder=3,
    visibility=0.65,
    beta_init=0.025,
    mortality_init=2,
    simulations_to_avg=48,
    num_of_shots=10,
)

df = TuningModelParams.get_tuning_results()
[print(item['fname']) for _, item in df.iterrows()]
#  ***********************************************************

# [15]  ******************************************************

#  ***********************************************************

# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************v
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************
# [15]  ******************************************************

#  ***********************************************************



