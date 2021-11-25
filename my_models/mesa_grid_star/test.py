import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import cProfile
import pstats
from collections import OrderedDict
import os
from multiprocessing import Process

from my_math_utils import *


class ClassA:
    def __init__(self, val1, val2, val3, N):
        self.atr1 = val1
        self.atr2 = val2
        self.atr3 = val3
        self.N = N

    def do_something(self):
        print("Do sth method started")
        s = 0
        for i in range(self.N):
            s += np.random.rand()
        print(f"Do sth method completed returns {s}")
        return s


class ClassB:
    def __init__(self, N):
        self.N = N
        self.agents = np.empty(N, dtype=ClassA)


@ njit
def fun(N, array_to_fill):
    for i in range(N):
        for j in range(2):
            if j == 0:
                array_to_fill[i][j] = i
            else:
                array_to_fill[i][j] = i*i*i

    return array_to_fill


@ njit
def test_function(N):
    matches = 0
    for day in range(N):
        shopping_days = np.random.randint(0, 7, 2)
        if day%7 in shopping_days:
            matches += 1


@ njit
def test_function2(N):
    matches = 0
    for day in range(N):
        shopping_days = np.random.randint(0, 7, 2)
        if day % 7 == shopping_days[0] or day % 7 == shopping_days[1]:
            matches += 1
        if day == N-1:
            print(matches)


if __name__ == '__main__':
    tup1 = ('a', 1, 3, 4)
    tup2 = ('a', 1, 5, 6)
    tup2 = ('b', 54, 25, 7)

    result = group_tuples_by_start(list_of_tuples=[tup1, tup2], start_length=1)
    print(result)


# COMMENTS ONLY *****************************************************************************************************

# iterations = 1000000
# test_function(10)
# test_function2(10)
#
#
# with cProfile.Profile() as pr:
# 	# test_function(N=iterations)
# 	# test_function2(N=iterations)
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats(10)



# def decide_who_does_shopping(self):
# 	for household_id in range(self.num_of_households):
# 		household_members = [agent for agent in self.schedule.agents if type(agent) is
# 		                     OrdinaryPearsonAgent and agent.household_id == household_id]
# 		if get_current_day(model=self) in household_members[0].shopping_days:
# 			set_which_household_member_do_shopping(household_members=household_members)


# def try_to_get_infected_by_cashier(self):
# 	if self.state == "susceptible":
# 		cellmates = self.model.grid.get_cell_list_contents([self.pos])
# 		cashier_cellmates = [agent for agent in cellmates if type(agent) is CashierAgent]
# 		for cashier in cashier_cellmates:
# 			if cashier.state == "prodromal" or cashier.state == 'illness':
# 				if random.uniform(0, 1) <= self.model.beta:
# 					self.go_into_incubation_phase()
# 					self.became_infected_today = True


# def try_to_infect_cashier(self):
# 	if self.state == "prodromal":
# 		cellmates = self.model.grid.get_cell_list_contents([self.pos])
# 		cashier_cellmates = [agent for agent in cellmates if type(agent) is CashierAgent]
# 		for cashier in cashier_cellmates:
# 			if cashier.state == "susceptible":
# 				if random.uniform(0, 1) <= self.model.beta:
# 					cashier.go_into_incubation_phase()