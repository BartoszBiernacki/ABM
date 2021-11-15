import cProfile
import pstats
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mesa.batchrunner import BatchRunner
from disease_model import DiseaseModel
from disease_server import server
from my_math_utils import group_tuples_by_start, fit_exp_to_peaks


# Returns dict in which keys are tuples in which entries are values of variable parameters and
# items are averaged model parameters
def run_simulation(variable_params, fixed_params, visualisation, multi, profiling, iterations, max_steps,
                   modified_brMP=False):
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


def find_tau_for_given_Ns_and_betas(Ns, betas, iterations, max_steps, plot_exp_fittings=True,
                                    plot_tau_vs_beta_for_each_N=True, modified_brMP=False, random_activation=False):
    # BATCH RUNNER SETTINGS---------------------------------------------------------------------------------------------
    fixed_params = {"width": 1,
                    "height": 1,
                    "num_of_customers_in_household": 1,
                    "num_of_cashiers_in_neighbourhood": 1,
                    "avg_incubation_period": 5,
                    "avg_prodromal_period": 3,
                    "avg_illness_period": 15,
                    "mortality": 0.0,
                    "initial_infection_probability": 0.7,
                    "start_with_infected_cashiers_only": True,
                    "random_activation": random_activation,
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
