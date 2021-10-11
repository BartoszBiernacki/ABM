from disease_model import DiseaseModel
from mesa.visualization.modules import CanvasGrid  # to show our grid type implemented in Model
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from disease_agent import PearsonAgent, DoctorAgent


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    # if type(agent) is PearsonAgent:
    #     if agent.infected:
    #         portrayal["Shape"] = "resources/sick.png"
    #         portrayal["scale"] = 0.9
    #         portrayal["Layer"] = 2
    #     elif agent.immune:
    #         portrayal["Shape"] = "resources/happy.png"
    #         portrayal["scale"] = 0.9
    #         portrayal["Layer"] = 1
    #     else:
    #         portrayal["Shape"] = "resources/idle.png"
    #         portrayal["scale"] = 0.9
    #         portrayal["Layer"] = 0
    # if type(agent) is DoctorAgent:
    #     portrayal["Shape"] = "resources/doctor.png"
    #     portrayal["scale"] = 0.5
    #     portrayal["Layer"] = 3
    # return portrayal

    if type(agent) is PearsonAgent:
        if agent.infected:
            portrayal["Layer"] = 2
            portrayal["Color"] = "Red"
            portrayal["r"] = 0.2
        elif agent.immune:
            portrayal["Color"] = "Blue"
            portrayal["r"] = 0.4
            portrayal["Layer"] = 1
        else:
            portrayal["Color"] = "Green"
            portrayal["Layer"] = 0
    if type(agent) is DoctorAgent:
        portrayal["Shape"] = "resources/doctor.png"
        portrayal["scale"] = 0.6
        portrayal["Layer"] = 3
    return portrayal


grid = CanvasGrid(agent_portrayal, grid_width=10, grid_height=10,
                  canvas_width=500, canvas_height=500)

total_infected_graph = ChartModule(
    series=[{"Label": "Total_pearson_infected", "Color": "Red"}],
    data_collector_name='datacollector')


number_of_pearson_agents_slider = UserSettableParameter('slider', "Number of pearson agents",
                                                        value=20, min_value=2, max_value=100,
                                                        step=1)
initial_infection_slider = UserSettableParameter('slider', "Probability of initial infection",
                                                 value=0.3, min_value=0.01, max_value=1, step=0.01)
pearson_level_of_movement_slider = UserSettableParameter('slider', "Person's level of movement",
                                                         value=0.2, min_value=0.01, max_value=1,
                                                         step=0.01)
transmissibility_slider = UserSettableParameter('slider', "Transmissibility",
                                                value=0.5, min_value=0.01, max_value=1, step=0.01)
mean_length_of_disease_slider = UserSettableParameter('slider', "Mean length of disease (days)",
                                                      value=10, min_value=1, max_value=100,
                                                      step=1)
immune_probability_slider = UserSettableParameter('slider',
                                                  "Probability to immune others by infected agent",
                                                  value=0.1, min_value=0.01, max_value=1,
                                                  step=0.01)
mean_length_of_immune_slider = UserSettableParameter('slider', "Mean length of immune (days)",
                                                     value=10, min_value=1, max_value=100,
                                                     step=1)
number_of_doctors_slider = UserSettableParameter('slider', "Number of doctor agents",
                                                 value=5, min_value=0, max_value=100, step=1)
doctor_level_of_movement_slider = UserSettableParameter('slider', "Doctor's level of movement",
                                                        value=0.2, min_value=0.01, max_value=1,
                                                        step=0.01)

server = ModularServer(model_cls=DiseaseModel,
                       visualization_elements=[grid, total_infected_graph],
                       name="Disease spread model", model_params=
                       {"num_pearson": number_of_pearson_agents_slider, "width": 10, "height": 10,
                        "initial_infection": initial_infection_slider,
                        "transmissibility": transmissibility_slider,
                        "pearson_level_of_movement": pearson_level_of_movement_slider,
                        "mean_length_of_disease": mean_length_of_disease_slider,
                        "immune_probability": immune_probability_slider,
                        "mean_length_of_immune": mean_length_of_immune_slider,
                        "num_doctors": number_of_doctors_slider,
                        "doctor_level_of_movement": doctor_level_of_movement_slider
                        })
