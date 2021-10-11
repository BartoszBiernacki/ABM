from mesa import Agent
import random


class PearsonAgent(Agent):
    def __init__(self, unique_id, model, initial_infection, transmissibility,
                 level_of_movement, mean_length_of_disease,
                 immune_probability, mean_length_of_immune):
        super().__init__(unique_id, model)
        self.transmissibility = transmissibility
        self.level_of_movement = level_of_movement
        self.mean_length_of_disease = mean_length_of_disease
        self.immune_probability = immune_probability
        self.mean_length_of_immune = mean_length_of_immune
        self.immune = False
        self.immune_duration = 0

        # initial infection with given probability
        if random.uniform(0, 1) <= initial_infection:
            self.infected = True
            # in exp distribution:
            # E(lambda) = 1/lambda ==> E(1/lambda)
            self.disease_duration = int(round(
                random.expovariate(1 / self.mean_length_of_disease), 0))
        else:
            self.infected = False

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True,
                                                          include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def infect_others(self):
        if self.infected:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            ordinary_cellmates = [agent for agent in cellmates if type(agent) is PearsonAgent]
            if len(ordinary_cellmates) > 1:
                for inhabitant in ordinary_cellmates:
                    if not inhabitant.infected:
                        if not inhabitant.immune:
                            if random.uniform(0, 1) < self.transmissibility:
                                inhabitant.infected = True
                                inhabitant.disease_duration = int(round(
                                    random.expovariate(1 / inhabitant.mean_length_of_disease), 0))

    # healthy agent can gain immunity to disease by sharing cell with infected (sick) agent if
    # firstly - sick agent will not infect him, secondly - has enough luck to get immunity
    def immune_others(self):
        if self.infected:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            ordinary_cellmates = [agent for agent in cellmates if type(agent) is PearsonAgent]
            if len(ordinary_cellmates) > 1:
                for inhabitant in ordinary_cellmates:
                    if not inhabitant.infected:
                        if random.uniform(0, 1) < inhabitant.immune_probability:
                            inhabitant.immune = True
                            inhabitant.immune_duration = int(round(
                                random.expovariate(1 / inhabitant.mean_length_of_immune), 0))

    def cure_immediately(self):
        if self.infected:
            self.infected = False
            self.disease_duration = 0

    def step(self):
        if random.uniform(0, 1) <= self.level_of_movement:
            self.move()

        if self.infected:
            self.infect_others()
            self.disease_duration -= 1
            if self.disease_duration <= 0:
                self.infected = False

        if self.infected:
            self.immune_others()

        if self.immune:
            self.immune_duration -= 1
            if self.immune_duration <= 0:
                self.immune = False


class DoctorAgent(Agent):
    def __init__(self, unique_id, model, level_of_movement):
        super().__init__(unique_id=unique_id, model=model)
        self.level_of_movement = level_of_movement

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def cure_ordinary_agent(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        ordinary_cellmates = [cellmate for cellmate in cellmates if type(cellmate) is PearsonAgent]
        ordinary_infected_agents = [agent for agent in ordinary_cellmates if agent.infected]
        if len(ordinary_infected_agents) > 0:
            agent_to_cure = random.choice(ordinary_infected_agents)
            agent_to_cure.cure_immediately()

    def step(self):
        if random.uniform(0, 1) <= self.level_of_movement:
            self.move()
        self.cure_ordinary_agent()
