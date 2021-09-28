from mesa_geo.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa_geo.visualization.MapModule import MapModule
from PerasonAgent import PersonAgent
from InfectedModel import InfectedModel

from my_utils import *


class InfectedText(TextElement):
    """ Display a text count of how many steps have been taken """
    def render(self, model):
        return "Steps: " + str(model.steps)


def portrayal_agents(agent):
    """Portrayal Method for canvas"""
    portrayal = dict()
    if isinstance(agent, PersonAgent):
        portrayal["radius"] = "3"

    if agent.atype in ["hotspot", "infected"]:
        portrayal["color"] = "Red"
    elif agent.atype in ["safe", "susceptible"]:
        portrayal["color"] = "Green"
    elif agent.atype in ["recovered"]:
        portrayal["color"] = "Blue"
    elif agent.atype in ["dead"]:
        portrayal["color"] = "Black"
    return portrayal


model_params = {
    "pop_size": UserSettableParameter("slider", "Population size", 50, 1, 100, 1),
    "init_infected": UserSettableParameter("slider", "Initial infected", 0.5, 0, 1, 0.01),
    "average_travelled_distance": UserSettableParameter("slider", "Average travelled distance", 50*kilometer, 0, 300*kilometer, kilometer),
    "infection_distance": UserSettableParameter("slider", "Average distance in which agent can infect", 40*kilometer, 0, 2000*kilometer, kilometer),
    "mean_length_of_disease": UserSettableParameter("slider", "Mean length of disease", 10, 1, 100, 1),
    "death_risk": UserSettableParameter("slider", "Death risk per day", 0.1, 0, 1, 0.01),
    "transmissibility": UserSettableParameter("slider", "Transmissibility", 0.2, 0, 1, 0.01),
    "immunity": UserSettableParameter("slider", "Immunity", 0.1, 0, 1, 0.01),
    "average_time_of_full_immunity_after_recovery": UserSettableParameter("slider", "Average time of full immunity after recovery", 3, 0, 100, 1)}

infected_text = InfectedText()

map_element = MapModule(portrayal_method=portrayal_agents, view=InfectedModel.MAP_COORDS,
                        zoom=6, map_height=500, map_width=600)

# names from datacolector in "InfectedModel" file
infected_chart = ChartModule(
    [
        {"Label": "infected", "Color": "Red"},
        {"Label": "susceptible", "Color": "Green"},
        {"Label": "recovered", "Color": "Blue"},
        {"Label": "dead", "Color": "Black"},
    ])

server = ModularServer(
    model_cls=InfectedModel,
    visualization_elements=[map_element, infected_text, infected_chart],
    name="Basic agent-based SIR model",
    model_params=model_params)

