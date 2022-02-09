from disease_spread_model.data_processing.text_processing import *
from disease_spread_model.config import Config
from disease_spread_model.model.model_runs import RunModel
from disease_spread_model.data_processing.real_data import RealData


class TuningModelParams(object):
    def __init__(self):
        pass
    
    # General usage method *************************************************************
    @classmethod
    def __get_prev_results(cls):
        try:
            df = pd.read_csv(Config.TUNING_MODEL_PARAMS_FNAME)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame(columns=[
                'voivodeship',
                'fit error per day',
                'lowest error',
                'visibility',
                'mortality',
                'beta',
                'runs',
                'first day',
                'last day',
                'shift',
                'timestamp',
                'fname'
            ])
        return df

    @classmethod
    def __save_tuning_result(cls, df_tuning_details: pd.DataFrame):
    
        def sort_df_and_mark_best_tuned_params(df):
            def mark_best(sub_df):
                # make sure that 'fit error per day' contains only floats
                sub_df['fit error per day'] = np.array(sub_df['fit error per day'], dtype=float)
            
                min_val = min(sub_df['fit error per day'])
                for idx, row in sub_df.iterrows():
                    if row['fit error per day'] == min_val:
                        sub_df.loc[idx, 'lowest error'] = True
                    else:
                        sub_df.loc[idx, 'lowest error'] = False
            
                return sub_df
        
            df = df.groupby('voivodeship').apply(mark_best)
            df.sort_values(by=['voivodeship', 'fit error per day'], inplace=True, ignore_index=True)
        
            return df
    
        def add_result_model_params_tuning_to_file(df_to_add: pd.DataFrame):
            import time
        
            df = cls.__get_prev_results()  # get df from file
            df_to_add['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.concat([df, df_to_add])  # insert row (merge)
        
            # sort df and mark best runs
            df = sort_df_and_mark_best_tuned_params(df)
        
            # convert floats to nicely formatted strings -------------
            cols_2f = ['fit error per day', 'visibility']
            cols_3f = ['mortality']
            cols_5f = ['beta']
        
            for col in cols_2f:
                df[col] = [f'{val:.2f}' for val in df[col]]
            for col in cols_3f:
                df[col] = [f'{val:.3f}' for val in df[col]]
            for col in cols_5f:
                df[col] = [f'{val:.5f}' for val in df[col]]
            # -------------------------------------------------------
        
            # save new df to csv
            save_dir = os.path.split(Config.TUNING_MODEL_PARAMS_FNAME)[0]
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(Config.TUNING_MODEL_PARAMS_FNAME, index=False)
    
        add_result_model_params_tuning_to_file(df_to_add=df_tuning_details)

    @classmethod
    def get_tuning_results(cls):
        return cls.__get_prev_results()

    @classmethod
    def _find_best_x_shift_to_match_plots(cls,
                                          y1_reference,
                                          y2,
                                          y2_start_idx,
                                          y2_end_idx):
        """
        Returns index of elem from which data y2[start: stop] best match any slice of the same length of y1.
        Also returns fit error (SSE).
        """
        y1_reference = np.array(y1_reference)
        y2 = np.array(y2[y2_start_idx: y2_end_idx + 1])
    
        smallest_difference = 1e9
        y2_length = y2_end_idx - y2_start_idx + 1
        shift = 0
    
        for i in range(len(y1_reference) - len(y2) + 1):
            y1_subset = y1_reference[i: i + y2_length]
        
            difference = np.sum((y2 - y1_subset) ** 2)
        
            if difference < smallest_difference:
                smallest_difference = difference
                shift = i
    
        shift = y2_start_idx - shift
    
        return shift, smallest_difference

    @classmethod
    def _find_shift_and_fit_error(cls,
                                  voivodeship: str,
                                  beta: float,
                                  mortality: float,
                                  visibility: float,
                                  iterations: int,
                                  starting_day: int,
                                  ending_day: int):
    
        """
        A general purpose method of this class.
        Runs simulation for given params and returns best
        shift and fit error.

        Runs simulations to evaluate how similar real data are to simulation data.
        Similarity is measured since beginning of pandemic for given voivodeship specified
        by percent_of_touched_counties. Real data are shifted among X axis to best match
        simulated data (only in specified interval).

        Function returns best shift and fit error associated with it.
        Fit error = SSE / days_to_fit_death_toll.
        """
    
        real_general_data = RealData.get_real_general_data()
        real_death_toll = RealData.get_real_death_toll()
    
        N = real_general_data.loc[voivodeship, 'N MODEL']
        grid_side_length = real_general_data.loc[voivodeship, 'grid side MODEL']
        grid_size = (grid_side_length, grid_side_length)
        infected_cashiers_at_start = grid_side_length
        max_steps = 250
    
        beta_sweep = (beta, 0.7, 1)
        beta_changes = ((1000, 2000), (1., 1.))
        mortality_sweep = (mortality / 100, 1., 1)
        visibility_sweep = (visibility, 1., 1)
    
        beta_mortality = [[(beta, mortality) for beta in np.linspace(*beta_sweep)]
                          for mortality in np.linspace(*mortality_sweep)]
    
        directory = RunModel.run_and_save_simulations(
            fixed_params={
                "grid_size": grid_size,
                "N": N,
                "customers_in_household": Config.customers_in_household,
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
        
            iterations=iterations,
            max_steps=max_steps,
        
            base_params=['grid_size',
                         'N',
                         'infected_cashiers_at_start',
                         'customers_in_household',
                         'infect_housemates_boolean'],
        
            remove_single_results=True
    
        )
    
        # As it ran simulations for only one triplet of (visibility, beta, mortality)
        # then result is the one file in avg_directory. To be specific it is
        # the latest file in directory, so read from it.
        fnames = all_fnames_from_dir(directory=directory)
        latest_file = max(fnames, key=os.path.getctime)
        df = pd.read_csv(latest_file)
    
        shift, error = cls._find_best_x_shift_to_match_plots(
            y1_reference=df['Dead people'],
            y2=real_death_toll.loc[voivodeship],
            y2_start_idx=starting_day,
            y2_end_idx=ending_day)
    
        df = pd.DataFrame(
            {
                "voivodeship": [voivodeship],
                "fit error per day": [error / (ending_day - starting_day)],
                "visibility": [visibility],
                "mortality": [mortality],
                "beta": [beta],
                "runs": [iterations],
                'first day': [starting_day],
                'last day': [ending_day],
                'shift': [shift],
                'fname': [latest_file]
            
            },
            index=[-1],
        )
    
        cls.__save_tuning_result(df_tuning_details=df)
        return shift, error / (ending_day - starting_day)

    # find beta for given voivodeships with other params fixed **************************
    @classmethod
    def _find_best_beta_for_given_mortality_visibility(cls,
                                                       voivodeship,
                                                       mortality,
                                                       visibility,
                                                       starting_day,
                                                       ending_day,
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

        def _find_shift_and_fit_error_for_beta(beta):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta,
                                                 mortality=mortality,
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)

        def _find_shift_and_fit_error_for_beta_minus(beta):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta * (1 - beta_scaling_factor),
                                                 mortality=mortality,
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)

        def _find_shift_and_fit_error_for_beta_plus(beta):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta * (1 + beta_scaling_factor),
                                                 mortality=mortality,
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)
        
        beta_val = beta_init
        
        best_shift, smallest_fit_error = _find_shift_and_fit_error_for_beta(beta=beta_val)
        print(f'For init beta={beta_val:.5f}, shift={best_shift}, error={smallest_fit_error:.2f}')
        
        shift_minus, error_minus = _find_shift_and_fit_error_for_beta_minus(beta=beta_val)
        print(f'For beta minus={beta_val * (1 - beta_scaling_factor):.5f}, shift={shift_minus}, error={error_minus:.2f}')
        
        shift_plus, error_plus = _find_shift_and_fit_error_for_beta_plus(beta=beta_val)
        print(f'For beta plus={beta_val * (1 + beta_scaling_factor):.5f}, shift={shift_plus}, error={error_plus:.2f}')
        
        improvement = False
        if error_minus < error_plus:
            which = 'minus'
            error = error_minus
            shift = shift_minus
            if error < smallest_fit_error:
                improvement = True
                best_shift = shift
                smallest_fit_error = error
                beta_val *= (1 - beta_scaling_factor)
            else:
                print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
                print(f"beta={beta_val:.5f}, error={smallest_fit_error:.2f}, shift={best_shift}")
        else:
            which = 'plus'
            error = error_plus
            shift = shift_plus
            if error < smallest_fit_error:
                improvement = True
                best_shift = shift
                smallest_fit_error = error
                beta_val *= (1 + beta_scaling_factor)
            else:
                print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
                print(f"beta={beta_val:.5f}, error={smallest_fit_error:.2f}, shift={best_shift}")
        
        if improvement:
            print(f'Beta {which} was better and will be further considered.')
            for i in range(fit_iterations):
                if which == 'minus':
                    test_shift, test_error = _find_shift_and_fit_error_for_beta_minus(beta=beta_val)
                    
                    if test_error < smallest_fit_error:
                        best_shift = test_shift
                        smallest_fit_error = test_error
                        beta_val *= (1 - beta_scaling_factor)
                    else:
                        break
                
                else:
                    test_shift, test_error = _find_shift_and_fit_error_for_beta_plus(beta=beta_val)
                    
                    if test_error < smallest_fit_error:
                        best_shift = test_shift
                        smallest_fit_error = test_error
                        beta_val *= (1 + beta_scaling_factor)
                    else:
                        break
                
                print(f'In step {i + 1} beta={beta_val:.5f}, shift={best_shift}, error {smallest_fit_error:.2f}')
            
            print(f"FINAL FIT for {voivodeship} with mortality={mortality}:")
            print(f"beta={beta_val:.5f}, error={smallest_fit_error:.2f}, shift={best_shift}")
            print()
            
            return beta_val, smallest_fit_error, best_shift
    
    @classmethod
    def find_beta_for_voivodeships(cls,
                                   voivodeships: list[str],
                                   starting_days: dict,
                                   ending_days: dict,
                                   mortality=2,
                                   visibility=0.65):
        
        """
        Runs simulations to fit beta for all voivodeship separately. Returns result dict


        :param voivodeships: list of voivodeships for which optimum beta will looked for.
            If voivodeships=['all'] it will be looked for all voivodeships.
        :type voivodeships: list[str]
        :param starting_days: dict like {voivodeship: num of the first day of pandemic}
        :type starting_days: dict
        :param ending_days: dict like {voivodeship: num of the last day of pandemic}
        :type ending_days: dict
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
        
        if 'all' in voivodeships:
            voivodeships = RealData.get_voivodeships()
        
        result = {}
        for voivodeship in voivodeships:
            fit = cls._find_best_beta_for_given_mortality_visibility(
                voivodeship=voivodeship,
                mortality=mortality,
                visibility=visibility,
                starting_day=starting_days[voivodeship],
                ending_day=ending_days[voivodeship],
                fit_iterations=8,
                beta_init=0.025
            )
            result[voivodeship] = fit
        return result

    # find mortality for given voivodeships with other params fixed **************************
    @classmethod
    def _find_best_mortality_for_given_beta_visibility(cls,
                                                       voivodeship,
                                                       beta,
                                                       visibility,
                                                       starting_day,
                                                       ending_day,
                                                       fit_iterations,
                                                       mortality_init=2,
                                                       mortality_scaling_factor=0.05,
                                                       ):
        """
        Runs multiple simulations with many mortalities until best is found.
        Goodness of mortality is measured by similarity in simulated and real (shifted) death toll.

        Returns tuple in which:
            first elem is best mortality value
            second elem is avg fit error per day
            third elem is vale of x shift to obtain best fit
        """
    
        def _find_shift_and_fit_error_for_mortality(mortality):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta,
                                                 mortality=mortality,
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)
    
        def _find_shift_and_fit_error_for_mortality_minus(mortality):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta,
                                                 mortality=mortality * (1 - mortality_scaling_factor),
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)
    
        def _find_shift_and_fit_error_for_mortality_plus(mortality):
            return cls._find_shift_and_fit_error(voivodeship=voivodeship,
                                                 beta=beta,
                                                 mortality=mortality * (1 + mortality_scaling_factor),
                                                 visibility=visibility,
                                                 iterations=12,
                                                 starting_day=starting_day,
                                                 ending_day=ending_day)
    
        mortality_val = mortality_init
    
        best_shift, smallest_fit_error = _find_shift_and_fit_error_for_mortality(mortality=mortality_val)
        print(f'For init mortality={mortality_val:.2f}, shift={best_shift}, error={smallest_fit_error:.2f}')
    
        shift_minus, error_minus = _find_shift_and_fit_error_for_mortality_minus(mortality=mortality_val)
        print(
            f'For mortality minus={mortality_val * (1 - mortality_scaling_factor):.2f}, shift={shift_minus}, '
            f'error={error_minus:.2f}')
    
        shift_plus, error_plus = _find_shift_and_fit_error_for_mortality_plus(mortality=mortality_val)
        print(f'For mortality plus={mortality_val * (1 + mortality_scaling_factor):.2f}, shift={shift_plus}, '
              f'error={error_plus:.2f}')
    
        improvement = False
        if error_minus < error_plus:
            which = 'minus'
            error = error_minus
            shift = shift_minus
            if error < smallest_fit_error:
                improvement = True
                best_shift = shift
                smallest_fit_error = error
                mortality_val *= (1 - mortality_scaling_factor)
            else:
                print(f"FINAL FIT for {voivodeship} with mortality={mortality_val}:")
                print(f"mortality={mortality_val:.2f}, error={smallest_fit_error:.2f}, shift={best_shift}")
        else:
            which = 'plus'
            error = error_plus
            shift = shift_plus
            if error < smallest_fit_error:
                improvement = True
                best_shift = shift
                smallest_fit_error = error
                mortality_val *= (1 + mortality_scaling_factor)
            else:
                print(f"FINAL FIT for {voivodeship} with mortality={mortality_val}:")
                print(f"beta={mortality_val:.2f}, error={smallest_fit_error:.2f}, shift={best_shift}")
    
        if improvement:
            print(f'Mortality {which} was better and will be further considered.')
            for i in range(fit_iterations):
                if which == 'minus':
                    test_shift, test_error = _find_shift_and_fit_error_for_mortality_minus(mortality=mortality_val)
                
                    if test_error < smallest_fit_error:
                        best_shift = test_shift
                        smallest_fit_error = test_error
                        mortality_val *= (1 - mortality_scaling_factor)
                    else:
                        break
            
                else:
                    test_shift, test_error = _find_shift_and_fit_error_for_mortality_plus(mortality=mortality_val)
                
                    if test_error < smallest_fit_error:
                        best_shift = test_shift
                        smallest_fit_error = test_error
                        mortality_val *= (1 + mortality_scaling_factor)
                    else:
                        break
            
                print(f'In step {i + 1} mortality={mortality_val:.2f}, shift={best_shift}, '
                      f'error {smallest_fit_error:.2f}')
        
            print(f"FINAL FIT for {voivodeship} with mortality={mortality_val}:")
            print(f"mortality={mortality_val:.2f}, error={smallest_fit_error:.2f}, shift={best_shift}")
            print()
        
            return mortality_val, smallest_fit_error, best_shift

    @classmethod
    def find_mortality_for_voivodeships(cls,
                                        voivodeships: list[str],
                                        starting_days: dict,
                                        ending_days: dict,
                                        beta=0.025,
                                        visibility=0.65):
    
        """
        Runs simulations to fit beta for all voivodeship separately. Returns result dict


        :param voivodeships: list of voivodeships for which optimum beta will looked for.
            If voivodeships=['all'] it will be looked for all voivodeships.
        :type voivodeships: list[str]
        :param starting_days: dict like {voivodeship: num of the first day of pandemic}
        :type starting_days: dict
        :param ending_days: dict like {voivodeship: num of the last day of pandemic}
        :type ending_days: dict
        :param beta: beta for which beta will be fitted (0.1 = 10%).
        :type beta: float
        :param visibility: visibility for which beta will be fitted (in percent).
        :type visibility: float
        :return: Returns fit results dict in which:
            key is voivodeship.
            value is tuple containing:
                [0] mortality.
                [1] fit error per day.
                [2] days shift to obtain best agreement between simulated and real data.
        """
    
        if 'all' in voivodeships:
            voivodeships = RealData.get_voivodeships()
    
        result = {}
        for voivodeship in voivodeships:
            fit = cls._find_best_mortality_for_given_beta_visibility(
                voivodeship=voivodeship,
                beta=beta,
                visibility=visibility,
                starting_day=starting_days[voivodeship],
                ending_day=ending_days[voivodeship],
                fit_iterations=8,
                mortality_init=2,
                mortality_scaling_factor=0.05
            )
            result[voivodeship] = fit
        return result
    



    


   
