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
* [02] Real number of prodromal days 3 or 4. For 4 it should be on average 55.93 prodromal peoples (matches Fig. 1).

# Some results
| N | iterations | BMP | maxiterationsperchild | RAM | time |
| --- | --- | --- | --- | --- | --- |
| 1000 | 10k | standard | N/A | 10.3 GB (65%) | 5 min 53 s |
| 1000 | 10k | modified | 100 | 8.5 GB (47%) | 6 min 28 s |
| 1000 | 20k | standard | N/A | 24.3 GB (94%) | 11 min 52 s |
| 1000 | 20k | modified | 200 |  |  |

# Problem
* It's monday 11:00 cashier was infected and is in incubation phase for 0h = 0 days.
* It's tuesday 11:00 cashier is in incubation phase for 24h = 1 day.
* It's wednesday 11:00 cashier is in incubation phase for 48h = 2 days.
* It's thursday 00:01 cashier is in which phase? Incubation duration: 61h = 2.55 days.

## Case 1
If we count with respect to beginning of new day, then the first reported moment in which cashier is in incubation
phase is tuesday 00:01 and cashier goes into prodromal phase on friday at 00:01. In that scenario 
incubation period is in fact bigger than reported due to last 13h from monday which
were counted as susceptible, but were incubation.

## Case 2
If we count with respect to the end of current day, then the first reported moment in which cashier is in incubation
phase is monday  23:59. New day begins. It's tuesday 00:01.
Should I decrease incubation period by 1 right now or wait for tuesday 23:59?
* If I decrease incubation period by 1 right now then cashier goes into prodromal phase on thursday at 00:01.
In that scenario incubation period is in fact smaller than reported due to first 11h from monday which
were counted as incubation, but were susceptible.

* If I decrease incubation period by 1 at 23:59 then cashier goes into prodromal phase on friday at 00:01.
In that scenario incubation period is in fact bigger than reported due to last 13h from monday which
were counted as susceptible, but were incubation.


## Case 3
Cashier goes into prodromal phase on thursday at 11:00.

### Counts with respect to the beginning of the day
Counted as susceptible on monday.
Counted as incubation on tuesday, wednesday, thursday.
Counted as prodromal on friday and so on.

### Counts with respect to the end of the day (currently used)
Counted as incubation on monday, tuesday, wednesday.
Counted as prodromal on thursday and so on.


## Case 4 
Cashier goes into prodromal phase on thursday at random hour.

### Counts with respect to the beginning of the day
Counted as susceptible on monday.
Counted as incubation on tuesday, wednesday, thursday.
Counted as prodromal on friday and so on.

### Counts with respect to the end of the day (currently used)
Counted as incubation on monday, tuesday, wednesday.
Counted as prodromal on thursday and so on.

