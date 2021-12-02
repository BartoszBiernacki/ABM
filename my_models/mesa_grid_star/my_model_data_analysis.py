import cProfile
import pstats
import os
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import warnings

from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
from disease_model import DiseaseModel
from disease_server import server
from my_math_utils import *


# Returns dict in which keys are tuples in which entries are values of variable parameters and
# items are averaged model parameters
def run_simulation(variable_params, fixed_params, visualisation, multi, profiling, iterations, max_steps,
                   return_details=False, show_avg_results=True):
    
    fixed_params['max_steps'] = max_steps
    if visualisation:
        server.port = 8521  # default
        server.launch()
    else:
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
                    data_collector_model = batch_run.get_collector_model()
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
                data_collector_model = batch_run.get_collector_model()
        else:
            if profiling:
                with cProfile.Profile() as pr:
                    batch_run = BatchRunner(model_cls=DiseaseModel,
                                            variable_parameters=variable_params,
                                            fixed_parameters=fixed_params,
                                            iterations=iterations,
                                            max_steps=max_steps)
                    batch_run.run_all()
                    data_collector_model = batch_run.get_collector_model()
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
                data_collector_model = batch_run.get_collector_model()
                
        avg_results = get_avg_results(data_collector_model_results=data_collector_model,
                                      variable_params=variable_params,
                                      show_avg_results=show_avg_results)
        
        if return_details:
            return avg_results, data_collector_model
        else:
            return avg_results


def get_avg_results(data_collector_model_results, variable_params, show_avg_results):
    # returns dict in which keys are tuples of variable_params and values are dataframes averaged over all iterations
    num_of_variable_model_params = len(variable_params)
    list_of_tuples = list(data_collector_model_results.keys())
    tuples_grouped = group_tuples_by_start(list_of_tuples=list_of_tuples, start_length=num_of_variable_model_params)

    result = {}
    for key in tuples_grouped.keys():  # key is a tuple by which other tuples were grouped. For example key=(5, 2, 2)
        lis = []
        for item in tuples_grouped[key]:  # items are full tuples. For example item=(5, 2, 2, ..., 0)
            lis.append(data_collector_model_results[item])  # list of results dataframes matching key_tuple=(5, 2, 2)
        array_with_all_iterations_results_for_specific_parameters = np.array(lis)

        average_array = np.mean(array_with_all_iterations_results_for_specific_parameters, axis=0)
        df = pd.DataFrame(data=average_array)
        df.columns = data_collector_model_results[list_of_tuples[0]].columns
        
        if show_avg_results:
            print(df.to_markdown())
        result[key] = df

    return result


def find_tau_for_given_Ns_and_betas(Ns, betas,
                                    avg_incubation_periods,
                                    incubation_period_bins,
                                    avg_prodromal_periods,
                                    prodromal_period_bins,
                                    iterations, max_steps,
                                    plot_exp_fittings=True,
                                    save_exp_fittings=True,
                                    plot_tau_vs_beta_for_each_N=True):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"width": 1,
                    "height": 1,
                    "num_of_customers_in_household": 1,
                    "avg_illness_period": 15,
                    "illness_period_bins": 1,
                    "mortality": 0.0,
                    "die_at_once": False,
                    "num_of_infected_cashiers_at_start": 1,
                    "infect_housemates_boolean": False,
                    "extra_shopping_boolean": True}

    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta": betas,
                       "avg_incubation_period": avg_incubation_periods,
                       "incubation_period_bins": incubation_period_bins,
                       "avg_prodromal_period": avg_prodromal_periods,
                       "prodromal_period_bins": prodromal_period_bins}
    # ---------------------------------------------------------------------------------------------------------------------

    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=False,
                             multi=True,
                             profiling=False,
                             iterations=iterations,
                             max_steps=max_steps)

    Ns = []
    betas = []
    taus = []
    for key in results.keys():
        avg_df = results[key]
        tau = fit_exp_to_peaks(x_data=avg_df["Day"], y_data=avg_df["Incubation people"],
                               plot=plot_exp_fittings,
                               save=save_exp_fittings,
                               N=key[0], beta=key[1],
                               exposed_period=key[2],
                               exposed_bins=key[3],
                               infected_period=key[4],
                               infected_bins=key[5],
                               show_details=True)

        Ns.append(key[0])
        betas.append(key[1])
        taus.append(tau)

        print(f"For N = {key[0]} and beta = {key[1]}, tau = {tau}")

    if plot_tau_vs_beta_for_each_N:
        df = pd.DataFrame(data=np.array([Ns, betas, taus]).T, columns=['N', 'beta', 'tau'])
        plot_tau_vs_beta_for_given_Ns(df=df)


def find_death_toll_for_given_Ns_betas_and_mortalities(Ns, betas, mortalities,
                                                       widths,
                                                       heights,
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

    variable_params = {"width": widths,
                       "height": heights,
                       "num_of_customers_in_household": nums_of_customer_in_household,
                       "num_of_households_in_neighbourhood": Ns,
                       "beta": betas,
                       "mortality": mortalities,
                       "num_of_infected_cashiers_at_start": num_of_infected_cashiers_at_start,
                       
                       "avg_incubation_period": avg_incubation_periods,
                       "incubation_period_bins": incubation_periods_bins,
                       "avg_prodromal_period": avg_prodromal_periods,
                       "prodromal_period_bins": prodromal_periods_bins,
                       "avg_illness_period": avg_illness_periods,
                       "illness_period_bins": illness_periods_bins
                       }
    # ------------------------------------------------------------------------------------------------------------------

    avg_results, detailed_results = run_simulation(variable_params=variable_params,
                                                   fixed_params=fixed_params,
                                                   visualisation=False,
                                                   multi=True,
                                                   profiling=False,
                                                   iterations=iterations,
                                                   max_steps=max_steps,
                                                   return_details=True,
                                                   show_avg_results=show_avg_results)
    
    for i, key in enumerate(avg_results.keys()):
        avg_df = avg_results[key]
        legend = f"Average data over {iterations} simulations"'\n'r"$\beta$"f"={key[4]}"'\n'f"mortality={key[5]*100}%"

        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.set_title(f"Grid = {key[0]} by {key[1]}\t\t"
                     f"Household size={key[2]}\n"
                     f"Households per neighbourhood={key[3]}\t\t"
                     f"Infected cashiers at start={key[6]}\n"
                     f"Exposed period={key[7]}"r"$\pm$"f"{key[8]//2}\t\t"
                     f"Infected period={key[9]}"r"$\pm$"f"{key[10]//2}\t\t"
                     f"Quarantine period={key[11]}"r"$\pm$"f"{key[12]//2}\t\t"
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
                            f"Beta={key[4]}"
                            f"_Mortality={key[5] * 100}_percent"
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
                            f"Beta={key[4]}"
                            f"_Mortality={key[5]*100}_percent"
                            f"_Plot_id={i}.pdf")
            plt.cla()
            # plt.show()
            
            
def find_fraction_of_susceptibles(Ns,
                                  betas,
                                  mortalities,
                                  widths,
                                  heights,
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
                                  show_avg_results=True
                                  ):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"die_at_once": False,
                    "extra_shopping_boolean": True,
                    'infect_housemates_boolean': infect_housemates}
    
    variable_params = {"width": widths,
                       "height": heights,
                       "num_of_customers_in_household": nums_of_customer_in_household,
                       "num_of_households_in_neighbourhood": Ns,
                       "beta": betas,
                       "mortality": mortalities,
                       "num_of_infected_cashiers_at_start": num_of_infected_cashiers_at_start,
    
                       "avg_incubation_period": avg_incubation_periods,
                       "incubation_period_bins": incubation_periods_bins,
                       "avg_prodromal_period": avg_prodromal_periods,
                       "prodromal_period_bins": prodromal_periods_bins,
                       "avg_illness_period": avg_illness_periods,
                       "illness_period_bins": illness_periods_bins
                       }
    # ------------------------------------------------------------------------------------------------------------------
    
    avg_results = run_simulation(variable_params=variable_params,
                                 fixed_params=fixed_params,
                                 visualisation=False,
                                 multi=True,
                                 profiling=False,
                                 iterations=iterations,
                                 max_steps=max_steps,
                                 return_details=False,
                                 show_avg_results=show_avg_results)
    
    for key in avg_results.keys():
        width = key[0]
        height = key[1]
        num_of_customers_in_household = key[2]
        num_of_households_in_neighbourhood = key[3]
        beta = key[4]
        mortality = key[5]
        num_of_infected_cashiers_at_start = key[6]
        
        avg_df = avg_results[key]
        
        plot_fraction_of_susceptible(avg_df=avg_df,
                                     grid_size=(width, height),
                                     N=num_of_households_in_neighbourhood,
                                     beta=beta,
                                     num_of_infected_cashiers_at_start=num_of_infected_cashiers_at_start)
        
        
def plot_fraction_of_susceptible(avg_df,
                                 grid_size,
                                 N,
                                 beta,
                                 num_of_infected_cashiers_at_start):
    x = avg_df['Day']
    y1 = avg_df['Susceptible people'] / avg_df['Susceptible people'][0]
    y2 = avg_df['Susceptible cashiers'] / (grid_size[0]*grid_size[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(r"N = {:1d}"'\t' r"$\beta$ = {:.3f}"'\n'
                 r"Grid size = {:1d} by {:1d}"'      '"Number of infected cashiers at start={:1d}".
                 format(N, beta,
                        grid_size[0], grid_size[1], num_of_infected_cashiers_at_start))
    ax.set_xlabel('t, days', fontsize=12)
    ax.set_ylabel('Fraction of susceptibles', fontsize=12)
    
    ax.plot(x, y1, label="people", color='blue', linewidth=2)
    ax.plot(x, y2, label="cashiers", color='red', linewidth=2, linestyle='dashed')
    
    
    plt.show()
    
    
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


def run_test_simulation(Ns, betas, iterations, max_steps):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"width": 1,
                    "height": 1,
                    "num_of_customers_in_household": 1,
                    "avg_incubation_period": 5,
                    "incubation_period_bins": 1,
                    "avg_prodromal_period": 3,
                    "prodromal_period_bins": 1,
                    "avg_illness_period": 15,
                    "illness_period_bins": 1,
                    "mortality": 0.0,
                    "num_of_infected_cashiers_at_start": 1,
                    "die_at_once": False,
                    "extra_shopping_boolean": True}

    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta": betas}
    # ---------------------------------------------------------------------------------------------------------------------

    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=False,
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
        ax.plot(avg_df['Day'], 6*avg_df['Incubation customers'], marker='.', markersize=10, color='yellow', linewidth=0)
        ax.plot(avg_df['Day'], (7/2)*avg_df['Incubation customers'], marker='.', markersize=10, color='brown', linewidth=0)

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


def reproduce_plot_by_website(filename="data_extracted_from_fig1.csv"):
    df = pd.read_csv(filename, header=None)
    x_img = np.array(df.iloc[:, [0]]).flatten()
    y_img = np.array(df.iloc[:, [1]]).flatten()
    print(x_img.size)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x_img, y_img)

    x = [0, 4, 5, 9, 12, 13, 16, 17, 20, 21, 24, 25]
    y = [0, 56, 56, 13.5, 38.5, 38.5, 20.5, 19.2, 28.5, 29.4, 19.9, 18.3]
    ax.plot(x, y, color='red', linestyle='-')
    ax.scatter(x, y, color='red')

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 51, 5)
    minor_ticks = np.arange(0, 51, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1)
    ax.set(xlim=(-1, 26))

    # plt.show()

    x_int_indexes = np.array(np.linspace(0, len(x_img)-1, int(max(x_img))), dtype=int)

    x_int = np.array(x_img[x_int_indexes], dtype=int)
    y_int = y_img[x_int_indexes]

    df = pd.DataFrame(data=np.array([x_int, y_int]))
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)

    ax.plot(x_int, y_int, color='green', linestyle='-')
    plt.show()
    print(np.argmax(y_img))
    print(x_img[np.argmax(y_img)])

    print(x_int_indexes)
