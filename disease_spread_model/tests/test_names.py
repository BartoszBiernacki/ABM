import unittest
from disease_spread_model.names import TRANSLATE


class TestNames(unittest.TestCase):
    def test_to_short_dict(self):
        example_dict = {
            'Runs': 48,
            'g_size': (20, 20),
            'N': 701,
            'Infected_cashiers_at_start': 20,
            'customers_in_household': 3,
        }
        
        expected = {
            'runs': 48,
            'g_size': (20, 20),
            'n': 701,
            'inf_cash': 20,
            'h_size': 3,
        }
        
        result = TRANSLATE.to_short(example_dict)
        
        self.assertEqual(result, expected,
                         'TRANSLATE.to_short(dict)failed!')

    def test_to_short_list(self):
        example_list = [
            'Runs',
            'grid_size',
            'N',
            'Infected_cashiers_at_start',
            'customers_in_household',
        ]
    
        expected = [
            'runs',
            'g_size',
            'n',
            'inf_cash',
            'h_size',
        ]
    
        result = TRANSLATE.to_short(example_list)
    
        self.assertEqual(result, expected,
                         'TRANSLATE.to_short(list) failed!')

    def test_to_short_str(self):
        example_str = 'Infected_cashiers_at_start'
        expected = 'inf_cash'
    
        result = TRANSLATE.to_short(example_str)
    
        self.assertEqual(result, expected,
                         'TRANSLATE.to_short(str) failed!')

    def test_to_long_dict(self):
        example_dict = {
            'Runs': 48,
            'g_size': (20, 20),
            'N': 701,
            'Infected_cashiers_at_start': 20,
            'h_size': 3,
        }
    
        expected = {
            'runs': 48,
            'grid_size': (20, 20),
            'n': 701,
            'infected_cashiers_at_start': 20,
            'customers_in_household': 3,
        }
    
        result = TRANSLATE.to_long(example_dict)
    
        self.assertEqual(result, expected,
                         'TRANSLATE.to_long(dict) failed!')

    def test_to_long_list(self):
        example_dict = [
            'Runs',
            'g_size',
            'N',
            'Infected_cashiers_at_start',
            'h_size',
        ]
    
        expected = [
            'runs',
            'grid_size',
            'n',
            'infected_cashiers_at_start',
            'customers_in_household',
        ]
    
        result = TRANSLATE.to_long(example_dict)
    
        self.assertEqual(result, expected,
                         'TRANSLATE.to_long(list) failed!')

    def test_to_long_str(self):
        example_str = 'inf_cash'
        expected = 'infected_cashiers_at_start'
    
        result = TRANSLATE.to_long(example_str)
    
        self.assertEqual(result, expected,
                         'TRANSLATE.to_long(str) failed!')


if __name__ == '__main__':
    unittest.main()
