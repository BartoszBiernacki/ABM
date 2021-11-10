# Speed optimization
* [01] **Accessing an item from a list is about 12% faster than from a dict.**
* [02] **Accessing an item from a numpy array takes twice as long as from a list.**
* [03] **np.random.choice is 250 times slower than straightforward choosing item from a list of length 3.**
The difference increases as the list length increases.
*  [04]  **Ordered agent activation is approximately 25% faster than random activation.**

# Customer states
OrdinaryPearsonAgent.state
* -1 = "recovery"
* 0 = "susceptible"
* 1 = "incubation" it is the same as "exposed"
* 2 = "prodromal"
* 3 = "illness"
* 4 = "dead"

"recovery" = -1, for quick check if agent is health enough to go shopping: **agent.state <= 2**

# Notation
## BatchRunnerMP
### Result data type
Collecting data by BatchRunnerMP returns data as dictionary in which
key is a tuple that looks like (variable_params_values, fixed_params_values, num_of_iteration).
Order of items in _params is the same as given in BatchRunnerMP constructor.

Values in this dictionary are pandas dataframes containing indicated model parameters
for each day of simulation. 

# Questions
* [01] Order of activation first agents or cashiers or maybe agents from first neighbourhood then 
cashier from first neighbourhood then agents from second neighbourhood e.t.c.