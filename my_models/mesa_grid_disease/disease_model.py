from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from my_utils import *

import random

from disease_agent import PearsonAgent, DoctorAgent


class DiseaseModel(Model):
    def __init__(self, num_pearson, width, height, initial_infection, transmissibility,
                 pearson_level_of_movement, mean_length_of_disease,
                 immune_probability, mean_length_of_immune,
                 num_doctors, doctor_level_of_movement):
        self.running = True  # required for BatchRunner
        self.num_pearson = num_pearson
        self.num_doctors = num_doctors
        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.schedule = RandomActivation(self)
        self.current_id = 0
        for i in range(self.num_pearson):
            a = PearsonAgent(unique_id=self.next_id(), model=self,
                             initial_infection=initial_infection,
                             transmissibility=transmissibility,
                             level_of_movement=pearson_level_of_movement,
                             mean_length_of_disease=mean_length_of_disease,
                             immune_probability=immune_probability,
                             mean_length_of_immune=mean_length_of_immune)
            self.schedule.add(a)
            # noinspection PyBroadException
            try:
                start_cell = self.grid.find_empty()
                self.grid.place_agent(agent=a, pos=start_cell)
            except:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(agent=a, pos=(x, y))

        for i in range(self.num_doctors):
            a = DoctorAgent(unique_id=self.next_id(), model=self,
                            level_of_movement=doctor_level_of_movement)
            self.schedule.add(a)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent=a, pos=(x, y))

        self.datacollector = DataCollector(
            model_reporters={"Total_pearson_infected": calculate_pearson_number_infected})

    def step(self):
        self.schedule.step()
        self.datacollector.collect(model=self)
