# Functions needed for datacollector
def get_infected_count(model):
    return model.counts["infected"]


def get_susceptible_count(model):
    return model.counts["susceptible"]


def get_recovered_count(model):
    return model.counts["recovered"]


def get_dead_count(model):
    return model.counts["dead"]


# equatorial circumference of the Earth
earth_circumference = 40075016.686
kilometer = 1000
