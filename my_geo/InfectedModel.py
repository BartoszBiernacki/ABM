from mesa.datacollection import DataCollector
from mesa import Model
from mesa.time import BaseScheduler
from mesa_geo.geoagent import AgentCreator
from mesa_geo import GeoSpace
from shapely.geometry import Point
import random

from my_utils import *
from PerasonAgent import PersonAgent
from NeighbourhoodAgent import NeighbourhoodAgent


class InfectedModel(Model):
    # Geographical parameters for desired map
    MAP_COORDS = [52.1, 19.2]  # Poland [y, x] (in degrees)
    geojson_regions = "resources/Poland-wojewodztwa.geojson"
    unique_id = "name"

    def __init__(self, pop_size, init_infected, average_travelled_distance, infection_distance,
                 mean_length_of_disease, death_risk, transmissibility, immunity,
                 average_time_of_full_immunity_after_recovery,
                 average_immunity_gain_after_recovery):

        self.schedule = BaseScheduler(self)
        self.grid = GeoSpace()
        self.steps = 0
        self.counts = None
        self.reset_counts()

        # SIR model parameters
        self.pop_size = pop_size
        self.counts["susceptible"] = pop_size

        self.running = True
        self.datacollector = DataCollector(
            {
                "infected": get_infected_count,
                "susceptible": get_susceptible_count,
                "recovered": get_recovered_count,
                "dead": get_dead_count,
            }
        )

        # Set up the Neighbourhood patches for every region in file (add to schedule later)
        # In geoJSON file in features every region has it's own unique name e.g. "name = Świętokrzyskie" so I used it as unique_id of NeighbourhoodAgent
        AC = AgentCreator(NeighbourhoodAgent, {"model": self})
        neighbourhood_agents = AC.from_file(filename=self.geojson_regions, unique_id=self.unique_id)
        self.grid.add_agents(neighbourhood_agents)

        # Generate PersonAgent population
        # Generate random location, add agent to grid and scheduler
        ac_population = AgentCreator(PersonAgent,
                                     {"model": self,
                                      "init_infected": init_infected,
                                      "average_travelled_distance": average_travelled_distance,
                                      "infection_distance": infection_distance,
                                      "mean_length_of_disease": mean_length_of_disease,
                                      "death_risk": death_risk,
                                      "transmissibility": transmissibility,
                                      "immunity": immunity,
                                      "average_time_of_full_immunity_after_recovery": average_time_of_full_immunity_after_recovery,
                                      "average_immunity_gain_after_recovery": average_immunity_gain_after_recovery})
        for i in range(pop_size):
            this_neighbourhood = random.choice(neighbourhood_agents)
            # Region where agent starts
            center_x, center_y = this_neighbourhood.shape.centroid.coords.xy
            this_bounds = this_neighbourhood.shape.bounds
            spread_x = int(this_bounds[2] - this_bounds[0])  # Heuristic for agent spread in region
            spread_y = int(this_bounds[3] - this_bounds[1])
            this_x = center_x[0] + self.random.randint(0, spread_x) - spread_x / 2
            this_y = center_y[0] + self.random.randint(0, spread_y) - spread_y / 2
            this_person = ac_population.create_agent(Point(this_x, this_y), "P" + str(i))
            self.grid.add_agents(this_person)
            self.schedule.add(this_person)

        # Add the neighbourhood agents to schedule AFTER person agents,
        # to allow them to update their color by using BaseScheduler
        for agent in neighbourhood_agents:
            self.schedule.add(agent)

        self.datacollector.collect(self)

    def reset_counts(self):
        self.counts = {
            "susceptible": 0,
            "infected": 0,
            "recovered": 0,
            "dead": 0,
            "safe": 0,
            "hotspot": 0,
        }

    def step(self):
        """Run one step of the model."""
        self.steps += 1
        self.reset_counts()
        self.schedule.step()
        self.grid._recreate_rtree()  # Recalculate spatial tree, because agents are moving

        self.datacollector.collect(self)

        # Run until no one is infected
        if self.counts["infected"] == 0:
            self.running = False
