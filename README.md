# Agent-based modelling of COVID-19 pandemic in Poland
## Project description

Project implements [mesa](https://mesa.readthedocs.io/en/stable/index.html) package to estimate number of people   
infected by COVID-19 in Poland  in the earliest state of pandemic.

The goal is based on following steps:
1. Collecting geospatial data about Poland with respect to voivodeships, like:
   - urbanization factor
   - number of people per shop
   - voivodeship population
2. Collecting data on COVID-19 in Poland with respect to voivodeships and counties, like:
    - death toll
    - infected toll
3. Second step is to build [Evgeniy Khain](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.022313) 
model.
4. Adjust model to collected data, such death toll will match.
5. The final step is to draw conclusions and estimations about  
number of infected people at an early stage of a pandemic in Polish voivodeships.

Original idea comes from [Evgeniy Khain](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.022313)
and assumes two-level model of quarantine.  
On the smallest level people are grouped in a star-network structure.  
Pearson in the center  maintains large number of interactions between other people  
in that cluster. Number of interactions between other people is limited. On the second  
level clusters are placed on 2D lattice with nearest-neighbour interactions.  
It corresponds to real life situations, because  we can group people in smaller communities  
like inhabitants of a given village in which one pearson is a cashier in grocery store.  
Sometimes inhabitants of different communities interact with each other, but it's unlikely  
if we compare that to interaction within the same community. In that analogy inhabitants  
of village makes one star-network structure with a cashier as center. Villages can't move,  
and they are separated like spaces on a 2D the lattice grid.

## Installation
### Mesa
Mesa installation is straightforward
```shell
pip install mesa
```

## Running model
To run my model created in that framework go to [disease_spread_model](disease_spread_model) and  
run [run.py](disease_spread_model/run.py).

For more examples go to [mesa GitHub repository.](https://github.com/projectmesa/mesa)

