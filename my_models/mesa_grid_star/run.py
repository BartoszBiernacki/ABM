from my_model_data_analysis import find_tau_for_given_Ns_and_betas
from my_model_data_analysis import reproduce_plot_by_website
from my_model_data_analysis import run_test_simulation
from my_model_data_analysis import find_death_toll_for_given_Ns_betas_and_mortalities
from my_model_data_analysis import find_fraction_of_susceptibles
from disease_server import run_simulation_in_browser

if __name__ == '__main__':

    # run_simulation_in_browser()
    
    find_death_toll_for_given_Ns_betas_and_mortalities(Ns=[1000],
                                                       betas=[0.04],
                                                       mortalities=[1.15/100],

                                                       widths=[20],
                                                       heights=[20],
                                                       nums_of_customer_in_household=[3],
                                                       avg_incubation_periods=[5],
                                                       incubation_periods_bins=[3],
                                                       avg_prodromal_periods=[3],
                                                       prodromal_periods_bins=[3],
                                                       avg_illness_periods=[15],
                                                       illness_periods_bins=[1],

                                                       infect_housemates=False,
                                                       num_of_infected_cashiers_at_start=[20],

                                                       iterations=12,
                                                       max_steps=700,
                                                       plot_first_k=10,
                                                       show_avg_results=False)
    
    

    # find_tau_for_given_Ns_and_betas(Ns=[1000],
    #                                 betas=[0.05],
    #                                 avg_incubation_periods=[5],
    #                                 incubation_period_bins=[1],
    #                                 avg_prodromal_periods=[3],
    #                                 prodromal_period_bins=[1],
    #                                 iterations=1000,
    #                                 max_steps=200,
    #                                 plot_exp_fittings=True,
    #                                 save_exp_fittings=False,
    #                                 plot_tau_vs_beta_for_each_N=False)

    # find_fraction_of_susceptibles(Ns=[1000],
    #                               betas=[0.07],
    #                               mortalities=[0.3 / 100],
    #
    #                               widths=[20],
    #                               heights=[20],
    #                               nums_of_customer_in_household=[3],
    #                               avg_incubation_periods=[5],
    #                               incubation_periods_bins=[3],
    #                               avg_prodromal_periods=[3],
    #                               prodromal_periods_bins=[3],
    #                               avg_illness_periods=[15],
    #                               illness_periods_bins=[1],
    #
    #                               num_of_infected_cashiers_at_start=[20],
    #
    #                               infect_housemates=False,
    #
    #                               iterations=24,
    #                               max_steps=500,
    #                               show_avg_results=True)

    # run_test_simulation(Ns=[1000],
    #                     betas=[0.05],
    #                     iterations=1000,
    #                     max_steps=30)

# reproduce_plot_by_website()
