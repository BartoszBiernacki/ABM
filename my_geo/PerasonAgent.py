import math
import numpy
from mesa_geo.geoagent import GeoAgent
from shapely.geometry import Point
import random
from NeighbourhoodAgent import NeighbourhoodAgent
from my_utils import *


def make_sick(agent):
    agent.atype = "infected"
    agent.disease_duration = int(round(random.expovariate(1 / agent.mean_length_of_disease), 0))


def make_recovered(agent):
    agent.atype = "recovered"
    agent.immune_duration = int(
        round(random.expovariate(1 / agent.average_time_of_full_immunity_after_recovery), 0))

# TODO implement gaining immunity after recovery
def make_susceptible_after_recovery(agent):
    agent.atype = "susceptible"


class PersonAgent(GeoAgent):
    """Person Agent."""

    def __init__(self, unique_id, model, shape, agent_type="susceptible",
                 average_travelled_distance=50 * kilometer,
                 infection_distance=40 * kilometer,
                 mean_length_of_disease=10,
                 death_risk=0.1, init_infected=0.1,
                 transmissibility=0.1, immunity=0.2,
                 average_time_of_full_immunity_after_recovery=5):  # in n days after recovery Pearson can't be infected again

        super().__init__(unique_id, model, shape)
        # Agent parameters
        self.atype = agent_type
        self.average_travelled_distance = average_travelled_distance
        self.infection_distance = infection_distance
        self.mean_length_of_disease = mean_length_of_disease
        self.death_risk = death_risk
        self.transmissibility = transmissibility
        self.immunity = immunity
        self.average_time_of_full_immunity_after_recovery = average_time_of_full_immunity_after_recovery

        self.disease_duration = 0
        self.immune_duration = 0

        # Random choose if infected and if so randomly set length of disease from exp distribution
        if self.random.random() < init_infected:
            make_sick(agent=self)
            self.model.counts["infected"] += 1  # Adjust initial counts
            self.model.counts["susceptible"] -= 1

    def infect_susceptible(self):
        if self.atype == "infected":
            neighbors = self.model.grid.get_neighbors_within_distance(self,
                                                                      self.infection_distance)
            for neighbor in neighbors:
                if neighbor.atype == "susceptible":
                    if random.random() < self.transmissibility:  # e.g how strong my cough is?
                        if random.random() > neighbor.immunity:  # e.g does other pearson wearing mask?
                            make_sick(neighbor)

    # Random move, r given by exp distribution, theta=rand(0, 2pi)
    def move_standard(self):
        curr_x = self.shape.x
        curr_y = self.shape.y

        r = random.expovariate(1 / self.average_travelled_distance)
        theta = random.uniform(0, 2 * numpy.pi)
        dx = r * numpy.cos(theta)
        dy = r * numpy.sin(theta)

        new_x = curr_x + dx
        new_y = curr_y + dy

        self.shape = Point(new_x, new_y)  # Reassign shape

    # move back in direction to region which we are care (to avoid travelling around the world)
    def move_in_direction_to_nearest_region(self):
        agents = self.model.schedule.agents
        regions = [agent for agent in agents if type(agent) is NeighbourhoodAgent]
        min_distance = earth_circumference
        nearest_region = None
        for region in regions:
            distance = self.model.grid.distance(self, region)
            if distance < min_distance:
                nearest_region = region
                min_distance = distance
        x_pos_of_center_of_nearest_region = nearest_region.shape.centroid.x
        y_pos_of_center_of_nearest_region = nearest_region.shape.centroid.y

        delta_y = y_pos_of_center_of_nearest_region - self.shape.y
        delta_x = x_pos_of_center_of_nearest_region - self.shape.x
        theta = math.atan2(delta_y, delta_x)
        theta = theta + random.uniform(-0.3, 0.3)  # add some randomness
        r = random.expovariate(1 / self.average_travelled_distance)

        dx = r * numpy.cos(theta)
        dy = r * numpy.sin(theta)

        new_x = self.shape.x + dx
        new_y = self.shape.y + dy

        self.shape = Point(new_x, new_y)  # Reassign shape

    def move(self):
        agents = self.model.schedule.agents
        regions = [agent for agent in agents if type(agent) is NeighbourhoodAgent]
        intersected = False
        for region in regions:
            if self.shape.intersection(region):
                intersected = True
                break
        if intersected:
            self.move_standard()
        else:
            self.move_in_direction_to_nearest_region()

    def step(self):
        """Advance one step."""
        # If infected, try to infect others (within given distance) and get better or die
        if self.atype == "infected":
            self.infect_susceptible()
            if self.random.random() < self.death_risk:
                self.atype = "dead"
            else:
                self.disease_duration -= 1
                if self.disease_duration <= 0:
                    make_recovered(self)
        elif self.atype == "recovered":
            self.immune_duration -= 1
            if self.immune_duration <= 0:
                make_susceptible_after_recovery(self)

        # If not dead, move
        if self.atype != "dead" and self.average_travelled_distance != 0:
            self.move()
        self.model.counts[self.atype] += 1  # Count agent type

    def __repr__(self):
        return "Person " + str(self.unique_id)
