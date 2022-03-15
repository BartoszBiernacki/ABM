import matplotlib.pyplot as plt
import numpy as np

from disease_spread_model.data_processing.real_data import RealData

if __name__ == '__main__':

    recovered_toll = RealData.recovered_toll()
    infected_toll = RealData.get_real_infected_toll()
    death_toll = RealData.get_real_death_toll()
    for voivodeship in RealData.get_voivodeships():
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        ax.plot(range(len(recovered_toll.columns)),
                recovered_toll.loc[voivodeship])

        ax.plot(range(len(infected_toll.columns)),
                infected_toll.loc[voivodeship])
        
        ax.plot(range(len(death_toll.columns)),
                death_toll.loc[voivodeship])

        ax2 = ax.twinx()

        max_idx = min(len(death_toll.columns), len(recovered_toll.columns))
        
        ax2.plot(range(len(recovered_toll.columns)),
                 recovered_toll.loc[voivodeship],
                 color='gray',
                 lw=2)
        
        # ax2.plot(range(len(death_toll.columns)),
        #          death_toll.loc[voivodeship],
        #          color='red', )
        
        
        plt.show()