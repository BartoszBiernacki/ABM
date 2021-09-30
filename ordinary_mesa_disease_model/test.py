# import the required libraries
import random
import matplotlib.pyplot as plt
import numpy as np

# store the random numbers in a list
nums = []
lambd = 3
sample = 100000

for i in range(sample):
    temp = random.expovariate(1/lambd)
    nums.append(temp)

# plotting a graph
his = plt.hist(nums, bins=1000)
print(his[0][0])
x = np.linspace(0, 10, 10000)
plt.plot(x, np.exp(-(1/lambd)*x)*his[0][0])
plt.xlim([0, 10])
plt.show()


