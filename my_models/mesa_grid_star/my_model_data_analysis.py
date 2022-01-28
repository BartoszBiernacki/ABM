import cProfile
import pstats

from mesa.batchrunner import BatchRunner
from mesa_batchrunner_modified import BatchRunnerMP     # saves results on disk to keep RAM free

from disease_model import DiseaseModel
from avg_results import *
from text_processing import *
from real_data import RealData
from config import *


def run_background_simulation(variable_params, fixed_params, multi, profiling, iterations, max_steps):
    
    fixed_params['max_steps'] = max_steps
    if multi:
        if profiling:
            with cProfile.Profile() as pr:
                batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                          nr_processes=os.cpu_count(),
                                          variable_parameters=variable_params,
                                          fixed_parameters=fixed_params,
                                          iterations=iterations,
                                          max_steps=max_steps)
                batch_run.run_all()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(10)
        else:
            batch_run = BatchRunnerMP(model_cls=DiseaseModel,
                                      nr_processes=os.cpu_count(),
                                      variable_parameters=variable_params,
                                      fixed_parameters=fixed_params,
                                      iterations=iterations,
                                      max_steps=max_steps)
            batch_run.run_all()
    else:
        if profiling:
            with cProfile.Profile() as pr:
                batch_run = BatchRunner(model_cls=DiseaseModel,
                                        variable_parameters=variable_params,
                                        fixed_parameters=fixed_params,
                                        iterations=iterations,
                                        max_steps=max_steps)
                batch_run.run_all()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(10)
        else:
            batch_run = BatchRunner(model_cls=DiseaseModel,
                                    variable_parameters=variable_params,
                                    fixed_parameters=fixed_params,
                                    iterations=iterations,
                                    max_steps=max_steps)
            batch_run.run_all()
            

def run_and_save_simulations(variable_params, fixed_params, save_dir,
                             iterations, max_steps,
                             base_params=None):
    
    run_background_simulation(variable_params=variable_params,
                              fixed_params=fixed_params,
                              multi=True,
                              profiling=False,
                              iterations=iterations,
                              max_steps=max_steps)
    
    results = get_avg_results(directory='TMP_SAVE/',
                              variable_params=variable_params,
                              ignore_dead_pandemics=True)

    directory = save_avg_results(avg_results=results,
                                 fixed_params=fixed_params,
                                 variable_params=variable_params,
                                 save_dir=save_dir,
                                 runs=iterations,
                                 base_params=base_params)

    # remove_tmp_results()
    
    return directory


def get_days_and_deaths_by_beta_and_fixed_mortality_from_dir(directory, const_mortality):
    fnames = all_fnames_from_dir(directory=directory)
    
    result = {}
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        mortality = float(variable_params['mortality'])
        if mortality == const_mortality:
            beta = variable_params['$\\beta$']
            result[beta] = df[['Day', 'Dead people']].copy()
        
    if not result:
        raise ValueError(f'Not found any data for const mortality = {const_mortality} in {directory}')
        
    return result


def get_days_and_deaths_by_mortality_and_fixed_beta_from_dir(directory, const_beta):
    fnames = all_fnames_from_dir(directory=directory)
    
    result = {}
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        beta = float(variable_params['$\\beta$'])
        if beta == const_beta:
            mortality = variable_params['mortality']
            result[mortality] = df[['Day', 'Dead people']].copy()
    
    if not result:
        raise ValueError(f'Not found any data for const mortality = {const_beta} in {directory}')
    
    return result


def find_shift_and_fit_error(voivodeship,
                             beta,
                             mortality,
                             visibility,
                             iterations,
                             percent_of_touched_counties,
                             days_to_fit_death_toll):
    """
    Runs simulations to evaluate how similar real data are to simulation data.
    Similarity is measured since beginning of pandemic for given voivodeship specified
    by percent_of_touched_counties. Real data are shifted among X axis to best match
    simulated data (only in specified interval).

    Function returns best shift and fit error associated with it.
    Fit error = SSE / days_to_fit_death_toll.
    """
    
    customers_in_household = 3
    real_data_obj = RealData(customers_in_household=customers_in_household)
    real_general_data = real_data_obj.get_real_general_data()
    real_death_toll = real_data_obj.get_real_death_toll()
    
    N = real_general_data.loc[voivodeship, 'N MODEL']
    grid_side_length = real_general_data.loc[voivodeship, 'grid side MODEL']
    grid_size = (grid_side_length, grid_side_length)
    infected_cashiers_at_start = grid_side_length
    max_steps = 250
    
    beta_sweep = (beta, 0.7, 1)
    beta_changes = ((1000, 2000), (1., 1.))
    mortality_sweep = (mortality/100, 1., 1)
    visibility_sweep = (visibility, 1., 1)
    
    beta_mortality = [[(beta, mortality) for beta in np.linspace(*beta_sweep)]
                      for mortality in np.linspace(*mortality_sweep)]
    
    directory = run_and_save_simulations(
        fixed_params={
            "grid_size": grid_size,
            "N": N,
            "customers_in_household": customers_in_household,
            "beta_changes": beta_changes,
            
            "avg_incubation_period": 5,
            "incubation_period_bins": 3,
            "avg_prodromal_period": 3,
            "prodromal_period_bins": 3,
            "avg_illness_period": 15,
            "illness_period_bins": 1,
            
            "die_at_once": False,
            "infected_cashiers_at_start": infected_cashiers_at_start,
            "infect_housemates_boolean": False,
            "extra_shopping_boolean": True
        },
        
        variable_params={
            "beta_mortality_pair": list(itertools.chain.from_iterable(beta_mortality)),
            "visibility": list(np.linspace(*visibility_sweep)),
        },
        
        save_dir='results/',
        iterations=iterations,
        max_steps=max_steps,
        
        base_params=['grid_size',
                     'N',
                     'infected_cashiers_at_start',
                     'customers_in_household',
                     'infect_housemates_boolean']
    )
    
    remove_tmp_results()
    
    starting_day = real_data_obj.get_starting_days_for_voivodeships_based_on_district_deaths(
        percent_of_touched_counties=percent_of_touched_counties,
        ignore_healthy_counties=True)
    
    fnames = all_fnames_from_dir(directory=directory)
    latest_file = max(fnames, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    shift, error = find_best_x_shift_to_match_plots(y1_reference=df['Dead people'],
                                                    y2=real_death_toll.loc[voivodeship],
                                                    y2_start=starting_day[voivodeship],
                                                    y2_end=starting_day[voivodeship] + days_to_fit_death_toll)
    
    return shift, error/days_to_fit_death_toll


def find_best_beta_for_given_mortality_visibility(voivodeship,
                                                  mortality,
                                                  visibility,
                                                  percent_of_touched_counties,
                                                  days_to_fit_death_toll,
                                                  fit_iterations,
                                                  beta_init=0.025,
                                                  beta_scaling_factor=0.05,
                                                  ):
    """
    Runs multiple simulations with many betas until best beta is found.
    Goodness of beta is measured by similarity in simulated and real (shifted) death toll.
    
    Returns tuple in which:
        first elem is best beta value
        second elem is avg fit error per day
        third elem is vale of x shift to obtain best fit
    """
    
    beta = beta_init
    
    best_shift, smallest_fit_error = find_shift_and_fit_error(voivodeship=voivodeship,
                                                              beta=beta,
                                                              mortality=mortality,
                                                              visibility=visibility,
                                                              iterations=12,
                                                              percent_of_touched_counties=percent_of_touched_counties,
                                                              days_to_fit_death_toll=days_to_fit_death_toll
                                                              )
    print(f'For init beta={beta}, shift={best_shift}, error={smallest_fit_error}')
    
    shift_minus, error_minus = find_shift_and_fit_error(voivodeship=voivodeship,
                                                        beta=beta * (1 - beta_scaling_factor),
                                                        mortality=mortality,
                                                        visibility=visibility,
                                                        iterations=12,
                                                        percent_of_touched_counties=percent_of_touched_counties,
                                                        days_to_fit_death_toll=days_to_fit_death_toll
                                                        )
    print(f'For beta minus={beta * (1 - beta_scaling_factor)}, shift={shift_minus}, error={error_minus}')
    
    shift_plus, error_plus = find_shift_and_fit_error(voivodeship=voivodeship,
                                                      beta=beta * (1 + beta_scaling_factor),
                                                      mortality=mortality,
                                                      visibility=visibility,
                                                      iterations=12,
                                                      percent_of_touched_counties=percent_of_touched_counties,
                                                      days_to_fit_death_toll=days_to_fit_death_toll
                                                      )
    print(f'For beta plus={beta * (1 + beta_scaling_factor)}, shift={shift_plus}, error={error_plus}')
    
    improvement = False
    if error_minus < error_plus:
        which = 'minus'
        error = error_minus
        shift = shift_minus
        if error < smallest_fit_error:
            improvement = True
            best_shift = shift
            smallest_fit_error = error
            beta = beta * (1 - beta_scaling_factor)
        else:
            print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
            print(f"beta={beta}, error={smallest_fit_error}, shift={best_shift}")
    else:
        which = 'plus'
        error = error_plus
        shift = shift_plus
        if error < smallest_fit_error:
            improvement = True
            best_shift = shift
            smallest_fit_error = error
            beta = beta * (1 + beta_scaling_factor)
        else:
            print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
            print(f"beta={beta}, error={smallest_fit_error}, shift={best_shift}")
    
    if improvement:
        print(f'Beta {which} was better and will be further considered.')
        for i in range(fit_iterations):
            if which == 'minus':
                test_shift, test_error = find_shift_and_fit_error(voivodeship=voivodeship,
                                                                  beta=beta * (1 - beta_scaling_factor),
                                                                  mortality=mortality,
                                                                  visibility=visibility,
                                                                  iterations=12,
                                                                  percent_of_touched_counties=percent_of_touched_counties,
                                                                  days_to_fit_death_toll=days_to_fit_death_toll
                                                                  )
                
                if test_error < smallest_fit_error:
                    best_shift = test_shift
                    smallest_fit_error = test_error
                    beta *= (1 - beta_scaling_factor)
                else:
                    break
            
            else:
                test_shift, test_error = find_shift_and_fit_error(voivodeship=voivodeship,
                                                                  beta=beta * (1 + beta_scaling_factor),
                                                                  mortality=mortality,
                                                                  visibility=visibility,
                                                                  iterations=12,
                                                                  percent_of_touched_counties=percent_of_touched_counties,
                                                                  days_to_fit_death_toll=days_to_fit_death_toll
                                                                  )
                if test_error < smallest_fit_error:
                    best_shift = test_shift
                    smallest_fit_error = test_error
                    beta *= (1 + beta_scaling_factor)
                else:
                    break
            
            print(f'In step {i+1} beta={beta}, shift={best_shift}, error {smallest_fit_error}')
        
        print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
        print(f"beta={beta}, error={smallest_fit_error}, shift={best_shift}")
        print()
        
        return beta, smallest_fit_error, best_shift


def find_beta_for_all_voivodeships(mortality=2,
                                   visibility=0.65):
    """
    Runs simulations to fit beta for all voivodeship separately. Returns result dict

    :param mortality: mortality for which beta will be fitted (in percent, 2=2%).
    :type mortality: float
    :param visibility: visibility for which beta will be fitted (in percent).
    :type visibility: float
    :return: Returns fit results dict in which:
        key is voivodeship.
        value is tuple containing:
            [0] beta.
            [1] fit error per day.
            [2] days shift to obtain best agreement between simulated and real data.
    """
    voivodeships = RealData.voivodeships
    percent_of_touched_counties = Config.percent_of_touched_counties
    days_to_fit_death_toll = Config.days_to_fit_death_toll
    
    result = {}
    for voivodeship in voivodeships:
        fit = find_best_beta_for_given_mortality_visibility(voivodeship=voivodeship,
                                                            mortality=mortality,
                                                            visibility=visibility,
                                                            percent_of_touched_counties=percent_of_touched_counties,
                                                            days_to_fit_death_toll=days_to_fit_death_toll,
                                                            fit_iterations=8,
                                                            beta_init=0.025
                                                            )
        result[voivodeship] = fit
    return result
