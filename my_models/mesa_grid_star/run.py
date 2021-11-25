from my_model_data_analysis import find_tau_for_given_Ns_and_betas
from my_model_data_analysis import reproduce_plot_by_website
from my_model_data_analysis import run_test_simulation
from disease_server import run_simulation_in_browser


if __name__ == '__main__':
    # run_simulation_in_browser()

    # find_tau_for_given_Ns_and_betas(Ns=[1000],
    #                                 betas=[0.05],
    #                                 iterations=100,
    #                                 max_steps=60,
    #                                 die_at_once=True,
    #                                 random_ordinary_pearson_activation=False,
    #                                 plot_exp_fittings=True,
    #                                 plot_tau_vs_beta_for_each_N=True,
    #                                 modified_brMP=False)

    # run_test_simulation(Ns=[1000],
    #                     betas=[0.05],
    #                     iterations=500,
    #                     max_steps=50,
    #                     die_at_once=True,
    #                     random_ordinary_pearson_activation=False,
    #                     modified_brMP=False)
    
    reproduce_plot_by_website()





























