def plot_fraction_of_susceptible(avg_df,
                                 grid_size,
                                 N,
                                 beta,
                                 num_of_infected_cashiers_at_start):
    x = avg_df['Day']
    y1 = avg_df['Susceptible people'] / avg_df['Susceptible people'][0]
    y2 = avg_df['Susceptible cashiers'] / (grid_size[0] * grid_size[1])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 1.1])
    
    ax.set_title(r"N = {:1d}"'\t' r"$\beta$ = {:.3f}"'\n'
                 r"Grid size = {:1d} by {:1d}"'      '"Number of infected cashiers at start={:1d}".
                 format(N, beta,
                        grid_size[0], grid_size[1], num_of_infected_cashiers_at_start))
    ax.set_xlabel('t, days', fontsize=12)
    ax.set_ylabel('Fraction of susceptibles', fontsize=12)
    
    ax.plot(x, y1, label="people", color='blue', linewidth=2)
    ax.plot(x, y2, label="cashiers", color='red', linewidth=2, linestyle='dashed')
    
    plt.show()


def find_death_toll_for_given_Ns_betas_and_mortalities(Ns,
                                                       beta_mortality_pairs,
                                                       grid_sizes,
                                                       num_of_infected_cashiers_at_start,
                                                       nums_of_customer_in_household,
                                                       avg_incubation_periods,
                                                       incubation_periods_bins,
                                                       avg_prodromal_periods,
                                                       prodromal_periods_bins,
                                                       avg_illness_periods,
                                                       illness_periods_bins,
                                                       infect_housemates,
                                                       iterations,
                                                       max_steps,
                                                       plot_first_k=0,
                                                       show_avg_results=True):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"die_at_once": False,
                    "extra_shopping_boolean": True,
                    'infect_housemates_boolean': infect_housemates}
    
    variable_params = {"grid_size": grid_sizes,
                       "num_of_customers_in_household": nums_of_customer_in_household,
                       "num_of_households_in_neighbourhood": Ns,
                       "beta_mortality_pair": beta_mortality_pairs,
                       "num_of_infected_cashiers_at_start": num_of_infected_cashiers_at_start,
    
                       "avg_incubation_period": avg_incubation_periods,
                       "incubation_period_bins": incubation_periods_bins,
                       "avg_prodromal_period": avg_prodromal_periods,
                       "prodromal_period_bins": prodromal_periods_bins,
                       "avg_illness_period": avg_illness_periods,
                       "illness_period_bins": illness_periods_bins
                       }
    # ------------------------------------------------------------------------------------------------------------------
    
    avg_results, detailed_results = run_background_simulation(variable_params=variable_params,
                                                              fixed_params=fixed_params,
                                                              multi=True,
                                                              profiling=False,
                                                              iterations=iterations,
                                                              max_steps=max_steps,
                                                              return_details=True,
                                                              show_avg_results=show_avg_results)
    
    for i, key in enumerate(avg_results.keys()):
        grid_size = key[0]
        nums_of_customers_in_household = key[1]
        N = key[2]
        beta = key[3][0]
        mortality = key[3][1]
        infected_cashiers_at_start = key[4]
        
        avg_incubation_period = key[5]
        incubation_period_bins = key[6]
        avg_prodromal_period = key[7]
        prodromal_period_bins = key[8]
        avg_illness_period = key[9]
        illness_period_bins = key[10]
        
        avg_df = avg_results[key]
        legend = f"Average data over {iterations} simulations"'\n'r"$\beta$"f"={beta}"'\n'f"mortality={mortality * 100}%"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.set_title(f"Grid = {grid_size}\t\t"
                     f"Household size={nums_of_customers_in_household}\n"
                     f"Households per neighbourhood={N}\t\t"
                     f"Infected cashiers at start={infected_cashiers_at_start}\n"
                     f"Exposed period={avg_incubation_period}"r"$\pm$"f"{incubation_period_bins // 2}\t\t"
                     f"Infected period={avg_prodromal_period}"r"$\pm$"f"{prodromal_period_bins // 2}\t\t"
                     f"Quarantine period={avg_illness_period}"r"$\pm$"f"{illness_period_bins // 2}\t\t"
                     )
        
        if plot_first_k == 0:
            ax.plot(avg_df["Day"], avg_df["Dead people"], label=legend, color='black')
            ax.set_xlabel('t, days', fontsize=20)
            ax.set_ylabel('Death toll', fontsize=20)
            ax.legend(loc='lower right', fontsize=15)
            
            # Save result as PDF =====================================================================================
            directory = "results/Death_toll_vs_days/"
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(f"{directory}"
                            f"Beta={beta}"
                            f"_Mortality={mortality * 100}_percent"
                            f"_Plot_id={i}.pdf")
            plt.cla()
        
        elif plot_first_k > 0:
            num_of_variable_model_params = len(variable_params)
            list_of_tuples = list(detailed_results.keys())
            tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples,
                                                   start_length=num_of_variable_model_params)
            
            lis = []
            for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
                lis.append(detailed_results[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
            
            array_with_all_iterations_results_for_specific_parameters = np.array(lis)
            
            ax.plot(avg_df["Day"], avg_df["Dead people"], label=legend, color='black', linewidth=3)
            ax.set_xlabel('t, days', fontsize=20)
            ax.set_ylabel('Death toll', fontsize=20)
            ax.legend(loc='lower right', fontsize=12)
            
            for iteration in range(min(plot_first_k, iterations)):
                df = pd.DataFrame(data=array_with_all_iterations_results_for_specific_parameters[iteration])
                df.columns = detailed_results[list_of_tuples[0]].columns
                
                ax.plot(df["Day"], df["Dead people"])
            
            # Save result as PDF =====================================================================================
            directory = "results/Death_toll_vs_days/"
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(f"{directory}"
                            f"Beta={beta}"
                            f"_Mortality={mortality * 100}_percent"
                            f"_Plot_id={i}.pdf")
            plt.cla()
            # plt.show()


def compare_death_tolls(
        Ns,
        beta_mortality_pairs,
        grid_sizes,
        num_of_infected_cashiers_at_start,
        nums_of_customer_in_household,
        avg_incubation_periods,
        incubation_periods_bins,
        avg_prodromal_periods,
        prodromal_periods_bins,
        avg_illness_periods,
        illness_periods_bins,
        infect_housemates,
        iterations,
        max_steps,
        plot_first_k=0,
        show_avg_results=True):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"die_at_once": False,
                    "extra_shopping_boolean": True,
                    'infect_housemates_boolean': infect_housemates}
    
    variable_params = {"grid_size": grid_sizes,
                       "num_of_customers_in_household": nums_of_customer_in_household,
                       "num_of_households_in_neighbourhood": Ns,
                       "beta_mortality_pair": beta_mortality_pairs,
                       "num_of_infected_cashiers_at_start": num_of_infected_cashiers_at_start,
    
                       "avg_incubation_period": avg_incubation_periods,
                       "incubation_period_bins": incubation_periods_bins,
                       "avg_prodromal_period": avg_prodromal_periods,
                       "prodromal_period_bins": prodromal_periods_bins,
                       "avg_illness_period": avg_illness_periods,
                       "illness_period_bins": illness_periods_bins
                       }
    # ------------------------------------------------------------------------------------------------------------------
    
    avg_results, detailed_results = run_background_simulation(variable_params=variable_params,
                                                              fixed_params=fixed_params,
                                                              multi=True,
                                                              profiling=False,
                                                              iterations=iterations,
                                                              max_steps=max_steps,
                                                              return_details=True,
                                                              show_avg_results=show_avg_results)
    
    for i, key in enumerate(avg_results.keys()):
        grid_size = key[0]
        nums_of_customers_in_household = key[1]
        N = key[2]
        beta = key[3][0]
        mortality = key[3][1]
        infected_cashiers_at_start = key[4]
        
        avg_incubation_period = key[5]
        incubation_period_bins = key[6]
        avg_prodromal_period = key[7]
        prodromal_period_bins = key[8]
        avg_illness_period = key[9]
        illness_period_bins = key[10]
        
        avg_df = avg_results[key]
        legend = f"Average data over {iterations} simulations"'\n'r"$\beta$"f"={beta}"'\n'f"mortality={mortality * 100}%"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.set_title(f"Grid = {grid_size}\t\t"
                     f"Household size={nums_of_customers_in_household}\n"
                     f"Households per neighbourhood={N}\t\t"
                     f"Infected cashiers at start={infected_cashiers_at_start}\n"
                     f"Exposed period={avg_incubation_period}"r"$\pm$"f"{incubation_period_bins // 2}\t\t"
                     f"Infected period={avg_prodromal_period}"r"$\pm$"f"{prodromal_period_bins // 2}\t\t"
                     f"Quarantine period={avg_illness_period}"r"$\pm$"f"{illness_period_bins // 2}\t\t"
                     )
        
        if plot_first_k == 0:
            ax.plot(avg_df["Day"], avg_df["Dead people"], label=legend, color='black')
            ax.set_xlabel('t, days', fontsize=20)
            ax.set_ylabel('Death toll', fontsize=20)
            ax.legend(loc='lower right', fontsize=15)
            
            # Save result as PDF =====================================================================================
            directory = "results/Death_toll_vs_days/"
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(f"{directory}"
                            f"Beta={beta}"
                            f"_Mortality={mortality * 100}_percent"
                            f"_Plot_id={i}.pdf")
            plt.cla()
        
        elif plot_first_k > 0:
            num_of_variable_model_params = len(variable_params)
            list_of_tuples = list(detailed_results.keys())
            tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples,
                                                   start_length=num_of_variable_model_params)
            
            lis = []
            for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
                lis.append(detailed_results[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
            
            array_with_all_iterations_results_for_specific_parameters = np.array(lis)
            
            ax.plot(avg_df["Day"], avg_df["Dead people"], label=legend, color='black', linewidth=3)
            ax.set_xlabel('t, days', fontsize=20)
            ax.set_ylabel('Death toll', fontsize=20)
            ax.legend(loc='lower right', fontsize=12)
            
            for iteration in range(min(plot_first_k, iterations)):
                df = pd.DataFrame(data=array_with_all_iterations_results_for_specific_parameters[iteration])
                df.columns = detailed_results[list_of_tuples[0]].columns
                
                ax.plot(df["Day"], df["Dead people"])
            
            # Save result as PDF =====================================================================================
            directory = "results/Death_toll_vs_days/"
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(f"{directory}"
                            f"Beta={beta}"
                            f"_Mortality={mortality * 100}_percent"
                            f"_Plot_id={i}.pdf")
            plt.cla()
    # plt.show()
    
    
def plot_tau_vs_beta_for_given_Ns(df):
    Ns = df['N'].unique()

    for N in Ns:
        filt = df['N'] == N

        ax = df[filt].plot(x='beta', y='tau', style='.-')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel(r'Transmission parameter $ \beta$', fontsize=12)
        ax.set_ylabel(r'$\tau$, days', fontsize=12)
        ax.legend(labels=[f"N={int(N)}"])
        plt.tight_layout()
        plt.show()


def run_test_simulation(Ns, beta_mortality_pairs, iterations, max_steps):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"grid_size": (1, 1),
                    "num_of_customers_in_household": 1,
    
                    "avg_incubation_period": 5,
                    "incubation_period_bins": 1,
                    "avg_prodromal_period": 3,
                    "prodromal_period_bins": 1,
                    "avg_illness_period": 15,
                    "illness_period_bins": 1,
    
                    "num_of_infected_cashiers_at_start": 1,
                    "die_at_once": False,
                    'infect_housemates_boolean': False,
                    "extra_shopping_boolean": True}
    
    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta_mortality_pair": beta_mortality_pairs}
    # ---------------------------------------------------------------------------------------------------------------------
    
    results = run_background_simulation(variable_params=variable_params,
                                        fixed_params=fixed_params,
                                        multi=True,
                                        profiling=False,
                                        iterations=iterations,
                                        max_steps=max_steps)
    
    for key in results.keys():
        avg_df = results[key]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        avg_df.plot(x='Day', y='Incubation people', marker='.', color='orange', ax=ax)
        avg_df.plot(x='Day', y='Incubation customers', marker='.', color='red', ax=ax)
        ax.plot(avg_df['Day'], 6 * avg_df['Incubation customers'], marker='.', markersize=10, color='yellow',
                linewidth=0)
        ax.plot(avg_df['Day'], (7 / 2) * avg_df['Incubation customers'], marker='.', markersize=10, color='brown',
                linewidth=0)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        avg_df.plot(x='Day', y='Prodromal people', marker='.', color='orange', ax=ax)
        avg_df.plot(x='Day', y='Prodromal customers', marker='.', color='red', ax=ax)
        ax.plot(avg_df['Day'], 6 * avg_df['Prodromal customers'], marker='.', markersize=10, color='yellow',
                linewidth=0)
        ax.plot(avg_df['Day'], (7 / 2) * avg_df['Prodromal customers'], marker='.', markersize=10, color='brown',
                linewidth=0)
        
        # print(avg_df['Incubation people'].divide(avg_df['Incubation customers']))
        # print(avg_df['Prodromal people'].divide(avg_df['Prodromal customers']))
        
        plt.show()


@njit(cache=True)
def find_out_who_wants_to_do_shopping(shopping_days_for_each_household_for_each_week,
                                      day,
                                      agents_state_grouped_by_households,
                                      A_ignore_quarantine_by_house_id,
                                      array_to_fill):
    total_num_of_households, num_of_shopping_days_in_week = array_to_fill.shape
    
    day_mod_7 = day % 7
    week = day // 7
    # return[household_id][0 or 1], if 0 --> volunteer_pos, if 1 --> volunteer_availability
    for household_id in range(total_num_of_households):
        if day_mod_7 in shopping_days_for_each_household_for_each_week[week][household_id]:
            
            # look up in healthy agents and in quarantined if is ignored
            volunteer_pos1 = np.argmax(agents_state_grouped_by_households[household_id] <= 2)
            volunteer_pos1_truth = False
            if agents_state_grouped_by_households[household_id][volunteer_pos1] <= 2:
                volunteer_pos1_truth = True
            
            volunteer_pos2 = \
                np.argmax(A_ignore_quarantine_by_house_id[household_id] * agents_state_grouped_by_households[
                    household_id] == 3)
            volunteer_pos2_truth = False
            if agents_state_grouped_by_households[household_id][volunteer_pos2] == 3 and \
                    A_ignore_quarantine_by_house_id[household_id][volunteer_pos2]:
                volunteer_pos2_truth = True
            
            # decide if goes from healthy or from quarantine ignorance
            if volunteer_pos1_truth and volunteer_pos2_truth:
                if random.uniform(a=0., b=1.) < 0.5:
                    array_to_fill[household_id][0] = volunteer_pos1
                    array_to_fill[household_id][1] = 1
                else:
                    array_to_fill[household_id][0] = volunteer_pos2
                    array_to_fill[household_id][1] = 1
            
            elif volunteer_pos1_truth:
                array_to_fill[household_id][0] = volunteer_pos1
                array_to_fill[household_id][1] = 1
            
            elif volunteer_pos2_truth:
                array_to_fill[household_id][0] = volunteer_pos2
                array_to_fill[household_id][1] = 1
            
            else:
                array_to_fill[household_id][0] = 0
                array_to_fill[household_id][1] = 0
        
        else:
            array_to_fill[household_id][0] = 0
            array_to_fill[household_id][1] = 0
    
    return array_to_fill


@njit(cache=True)
def neigh_id_of_extra_shopping(nearest_neighbourhoods_by_neigh_id, neigh_id_by_house_id, array_to_fill):
    # result[week][household_id] --> extra neighbourhood_id to visit
    
    max_weeks, total_households = array_to_fill.shape
    total_neighbourhoods, num_of_nearest_neighbourhoods = nearest_neighbourhoods_by_neigh_id.shape
    
    if num_of_nearest_neighbourhoods > 0:  # do sth if there is at least one neighbour
        
        covered_cycles = 0
        weeks_to_cover = max_weeks
        
        while True:
            for i in range(min(weeks_to_cover, num_of_nearest_neighbourhoods)):
                for house_id in range(total_households):
                    neigh_id = neigh_id_by_house_id[house_id]
                    shuffled_neighbours = np.random.permutation(nearest_neighbourhoods_by_neigh_id[neigh_id])
                    array_to_fill[covered_cycles * num_of_nearest_neighbourhoods + i][house_id] = shuffled_neighbours[i]
                    print(shuffled_neighbours[i])
            covered_cycles += 1
            weeks_to_cover -= num_of_nearest_neighbourhoods
            
            if weeks_to_cover <= 0:
                return array_to_fill


def plot_2D_pandemic_time(directory):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    df = pd.read_csv(fnames[0])
    print(df.head().to_markdown())
    
    betas = np.empty(len(fnames))
    mortalities = np.empty_like(betas)
    days = np.empty_like(betas)
    
    for i, fname in enumerate(fnames):
        variable_params = variable_params_from_fname(fname=fname)
        df = pd.read_csv(fname)
        
        filt = df['Dead people'] / np.max(df['Dead people']) > 0.99
        days[i] = np.min(df[filt]['Day'])
        
        betas[i] = variable_params['$\\beta$']
        mortalities[i] = float(variable_params['mortality']) * 100
    
    unique_betas = list(set(betas))
    unique_mortalities = list(set(mortalities))
    unique_betas.sort()
    unique_mortalities.sort()
    
    days_matrix = np.empty((len(unique_mortalities), len(unique_betas)))
    
    for i in range(len(fnames)):
        beta = betas[i]
        mortality = mortalities[i]
        
        days_matrix[unique_mortalities.index(mortality)][unique_betas.index(beta)] = days[i]
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    main_title = r"Pandemic duration in days as function of $\beta$ and mortality"'\n'
    title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
    ax.set_title(title)
    
    im = ax.pcolormesh(unique_betas, unique_mortalities, days_matrix, shading='nearest')
    ax.set_xticks(unique_betas[:])
    ax.set_yticks(unique_mortalities[:])
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel('Mortality (in percent)', fontsize=12)
    
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    for i in range(len(fnames)):
        ax.text(betas[i], mortalities[i], str(int(days[i])), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    plt.tight_layout()
    plt.show()


def plot_fraction_of_susceptible(fname):
    fixed_params = fixed_params_from_fname(fname=fname)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    main_title = r"Fraction of susceptible"'\n'
    title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
    ax.set_title(title)
    
    df = pd.read_csv(fname)
    variable_params = variable_params_from_fname(fname=fname)
    
    legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=None)
    
    ax.plot(df['Day'], df['Susceptible people'] / np.max(df['Susceptible people']), label=legend, color='blue')
    ax.plot(df['Day'], df['Susceptible cashiers'] / np.max(df['Replaced cashiers']), label=legend, color='red')
    
    ax.legend()
    
    plt.tight_layout()
    plt.show()