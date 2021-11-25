import cProfile
import pstats
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d

from mesa.batchrunner import BatchRunner
from disease_model import DiseaseModel
from disease_server import server
from my_math_utils import group_tuples_by_start, fit_exp_to_peaks


# Returns dict in which keys are tuples in which entries are values of variable parameters and
# items are averaged model parameters
def run_simulation(variable_params, fixed_params, visualisation, multi, profiling, iterations, max_steps,
                   modified_brMP=False):
    fixed_params['max_steps'] = max_steps
    if modified_brMP:
        from mesa_batchrunner_modified import BatchRunnerMP
    else:
        from mesa.batchrunner import BatchRunnerMP

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

        return get_avg_results(data_collector_model_results=data_collector_model, variable_params=variable_params)


def get_avg_results(data_collector_model_results, variable_params):
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
        print(df.to_markdown())
        result[key] = df

    return result


def find_tau_for_given_Ns_and_betas(Ns, betas, iterations, max_steps,
                                    die_at_once,
                                    random_ordinary_pearson_activation,
                                    plot_exp_fittings=True,
                                    plot_tau_vs_beta_for_each_N=True,
                                    modified_brMP=False):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"width": 1,
                    "height": 1,
                    "num_of_customers_in_household": 1,
                    "num_of_cashiers_in_neighbourhood": 1,
                    "avg_incubation_period": 5,
                    "avg_prodromal_period": 3,
                    "avg_illness_period": 15,
                    "mortality": 0.0,
                    "die_at_once": die_at_once,
                    "initial_infection_probability": 0.7,
                    "start_with_infected_cashiers_only": True,
                    "random_ordinary_pearson_activation": random_ordinary_pearson_activation,
                    "extra_shopping_boolean": False}

    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta": betas}
    # ---------------------------------------------------------------------------------------------------------------------

    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=False,
                             multi=True,
                             profiling=False,
                             iterations=iterations,
                             max_steps=max_steps,
                             modified_brMP=modified_brMP)

    Ns = []
    betas = []
    taus = []
    for key in results.keys():
        avg_df = results[key]
        tau = fit_exp_to_peaks(x_data=avg_df["Day"], y_data=avg_df["Incubation people"],
                               plot=plot_exp_fittings, N=key[0], beta=key[1])

        Ns.append(key[0])
        betas.append(key[1])
        taus.append(tau)

        print(f"For N = {key[0]} and beta = {key[1]}, tau = {tau}")

    df = pd.DataFrame(data=np.array([Ns, betas, taus]).T, columns=['N', 'beta', 'tau'])

    if plot_tau_vs_beta_for_each_N:
        plot_tau_vs_beta_for_given_Ns(df=df)


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


def run_test_simulation(Ns, betas, iterations, max_steps,
                            die_at_once,
                            random_ordinary_pearson_activation,
                            modified_brMP=False):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"width": 1,
                    "height": 1,
                    "num_of_customers_in_household": 1,
                    "num_of_cashiers_in_neighbourhood": 1,
                    "avg_incubation_period": 5,
                    "avg_prodromal_period": 3,
                    "avg_illness_period": 15,
                    "mortality": 0.0,
                    "die_at_once": die_at_once,
                    "initial_infection_probability": 0.7,
                    "start_with_infected_cashiers_only": True,
                    "random_ordinary_pearson_activation": random_ordinary_pearson_activation,
                    "extra_shopping_boolean": False}

    variable_params = {"num_of_households_in_neighbourhood": Ns,
                       "beta": betas}
    # ---------------------------------------------------------------------------------------------------------------------

    results = run_simulation(variable_params=variable_params,
                             fixed_params=fixed_params,
                             visualisation=False,
                             multi=True,
                             profiling=False,
                             iterations=iterations,
                             max_steps=max_steps,
                             modified_brMP=modified_brMP)

    for key in results.keys():
        avg_df = results[key]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        avg_df.plot(x='Day', y='Incubation people', marker='.', color='orange', ax=ax)
        avg_df.plot(x='Day', y='Incubation customers', marker='.', color='red', ax=ax)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        avg_df.plot(x='Day', y='Prodromal people', marker='.', color='yellow', ax=ax)
        avg_df.plot(x='Day', y='Prodromal customers', marker='.', color='red', ax=ax)

        print(avg_df['Incubation people'].divide(avg_df['Incubation customers']))
        print(avg_df['Prodromal people'].divide(avg_df['Prodromal customers']))

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


