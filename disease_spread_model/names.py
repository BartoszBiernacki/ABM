from typing import Union


class TRANSLATE:
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

    POLISH = {
        'Runs': 'Powtórzeń',
        'grid_size': 'siatka',
        'customers_in_household': 'rozmiar gospodarstwa',
        'n': 'N',
        'beta': r'$\beta$',
        'mortality': 'śmiertelność',
        'visibility': 'widoczność',
        'infected_cashiers_at_start':
            'początkowa ilość zainfekowanych kasjerów',
        'percent_of_infected_customers_at_start':
            'początkowy procent zainfekowanych klientów',
        'housemate_infection_probability':
            'prawdopodobieństwo zarażenia współlokatora',
    }

    LONG = {v: k for k, v in SHORT.items()}

    @classmethod
    def to_short(cls, params: Union[dict, list, tuple, set, str]):

        if isinstance(params, dict):
            return {(cls.SHORT[k.lower()] if k.lower() in cls.SHORT
                     else k.lower()): v
                    for k, v in params.items()}

        elif isinstance(params, (list, tuple, set)):
            translated = [(cls.SHORT[p.lower()] if p.lower() in cls.SHORT
                           else p.lower()) for p in params]
            return type(params)(translated)

        elif isinstance(params, str):
            return cls.to_short([params])[0]

        else:
            raise NotImplementedError

    @classmethod
    def to_long(cls, params: Union[dict, list, tuple, set, str]):

        if isinstance(params, dict):
            return {(cls.LONG[k.lower()] if k.lower() in cls.LONG
                     else k.lower()): v for k, v in params.items()}

        elif isinstance(params, (list, tuple, set)):
            translated = [(cls.LONG[p.lower()] if p.lower() in cls.LONG
                           else p.lower()) for p in params]
            return type(params)(translated)

        elif isinstance(params, str):
            return cls.to_long([params])[0]

        else:
            raise NotImplementedError

    @classmethod
    def to_polish(cls, params: Union[dict, list, tuple, set, str]):
        long = cls.to_long(params)

        if isinstance(long, dict):
            return {(cls.POLISH[k.lower()]
                     if k.lower() in cls.POLISH else k.lower()): v
                    for k, v in long.items()}

        elif isinstance(long, list):
            translated = [(cls.POLISH[p.lower()] if p.lower() in cls.POLISH
                           else p.lower()) for p in long]
            return type(params)(translated)

        elif isinstance(long, str):
            return cls.to_polish([long])[0]

        else:
            raise NotImplementedError
