# -*- coding: utf-8 -*-
"""
@Date : 2020/5/8 15:51
@File : DataForFitting.py
@author : SaltFish
@Tool : PyCharm

"""
import numpy as np
import matplotlib.pyplot as plt
import random

# Parameter settings
'''create dots surrounded by line"kx+b" '''
k = 0.5
b = 10
chaos_x = 5
chaos_y = 10
x = 0

# Create data
dots = []
for i in range(0, 100):
    x += abs(np.random.normal(0, chaos_x))
    y = x * k + b + np.random.normal(0, chaos_y)
    dots.append([x, y])


# Draw data
plt.scatter([x[0] for x in dots], [x[1] for x in dots], c='b')
plt.title('DataForFitting')

# Save data
## save pic
plt.savefig('../Data&Results/Fitting/pic.png')
## save data as txt file
random.shuffle(dots)
'''change data to random list'''
filename = 'data.txt'
with open('../Data&Results/Fitting/' + filename, 'w') as file:
    for dot in dots:
        file.write(str(dot[0]) + "," + str(dot[1]))
        file.write('\n')
plt.show()
