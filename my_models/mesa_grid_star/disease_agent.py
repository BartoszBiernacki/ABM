from mesa import Agent


# needed only to map neighbourhood xy to id and for batchrunner to work and to portrayal agent while visualisation
class CashierAgent(Agent):
    def __init__(self, unique_id, model, neighbourhood_id):
        super().__init__(unique_id, model)
        
        self.neighbourhood_id = neighbourhood_id
        self.state = model.C_state_by_neigh_id[self.neighbourhood_id]

    # Helpful agent functions =========================================================================================
    def show_neighbourhood(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        return self.pos, possible_steps

    def step(self):
        self.state = self.model.C_state_by_neigh_id[self.neighbourhood_id]
    
    