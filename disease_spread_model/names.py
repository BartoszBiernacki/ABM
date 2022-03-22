class TRANSLATE(object):
    SHORT = {
        'Runs': 'Runs',
        'grid_size': 'g_size',
        'customers_in_household': 'h_size',
        'n': 'n',
        'beta': 'b',
        'mortality': 'm',
        'visibility': 'v',
        'infected_cashiers_at_start': 'inf_cash',
        'percent_of_infected_customers_at_start': 'pct_inf_cust',
        'housemate_infection_probability': 'h_inf_prob',
    }
    
    LONG = {v: k for k, v in SHORT.items()}
    
    @classmethod
    def to_short(cls, params):
        
        if isinstance(params, dict):
            return {(cls.SHORT[k.lower()] if k.lower() in cls.SHORT else k.lower()): v
                    for k, v in params.items()}
        
        elif isinstance(params, list):
            return [(cls.SHORT[p.lower()] if p.lower() in cls.SHORT else p.lower())
                    for p in params]
        
        elif isinstance(params, str):
            return cls.to_short([params])[0]
        
        else:
            raise NotImplementedError

    @classmethod
    def to_long(cls, params):
        
        if isinstance(params, dict):
            return {(cls.LONG[k.lower()] if k.lower() in cls.LONG else k.lower()): v
                    for k, v in params.items()}
    
        elif isinstance(params, list):
            return [(cls.LONG[p.lower()] if p.lower() in cls.LONG else p.lower())
                    for p in params]
        
        elif isinstance(params, str):
            return cls.to_long([params])[0]
        
        else:
            raise NotImplementedError
