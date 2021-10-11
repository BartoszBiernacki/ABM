from disease_agent import PearsonAgent, DoctorAgent


def calculate_pearson_number_infected(model):
    pearson_agents = [agent for agent in model.schedule.agents if type(agent) is PearsonAgent]
    infection_report = [agent.infected for agent in pearson_agents if agent.infected]
    return len(infection_report)



