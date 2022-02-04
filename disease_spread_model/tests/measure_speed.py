from my_model_data_analysis import run_simulation


fixed_params = {"num_of_households_in_neighbourhood": 1000,
                "mortality": 0.01,
                "die_at_once": False,
                "width": 1,
                "height": 1,
                "num_of_customers_in_household": 1,
                "avg_incubation_period": 5,
                "incubation_period_bins": 1,
                "avg_prodromal_period": 4,
                "prodromal_period_bins": 1,
                "avg_illness_period": 15,
                "illness_period_bins": 1,
                "extra_shopping_boolean": False}


variable_params = {"beta": [0.05]}

run_simulation(variable_params=variable_params,
               fixed_params=fixed_params,
               visualisation=False,
               multi=False,
               profiling=True,
               iterations=100,
               max_steps=150)
