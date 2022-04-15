import pandas as pd
import pickle
from pathlib import Path
import os
import warnings
from abc import ABC, abstractmethod
from typing import Optional

# from disease_spread_model.data_processing.real_data import RealData
# from disease_spread_model.model.model_adjustment import OptimizeResultsWriterReader
# from disease_spread_model.run import voivodeship


class Root(ABC):
    def __init__(self):
        self.x = 'root'
        self.y = 1

    @abstractmethod
    def _set_y(self):
        pass

    @abstractmethod
    def run(self):
        pass


class Medium(Root, ABC):
    def __init__(self):
        super().__init__()

    def run(self):
        self._set_y()
        print(self.x, self.y)


class Child1(Medium):
    def __init__(self):
        super().__init__()
        self.x = 'child1'

    def _set_y(self):
        self.y = 3


c = Child1()
c.run()





