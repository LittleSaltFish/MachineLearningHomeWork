# -*- coding: utf-8 -*-
"""
@Date : 2020/5/8 15:51
@File : DataForClassify.py
@author : SaltFish
@Tool : PyCharm

"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Parameter settings
'''create dots surrounded by two centers'''
## set the center
center_a = [25, 50]
center_b = [50, 25]
## set the standard deviation
chaos_a = 5
chaos_b = 5

# Create data
dots = []
'''set of all dots'''
for i in range(0, 100):
    dot = [np.random.normal(center_a[0], chaos_a), np.random.normal(center_a[1], chaos_a)]
    dots.append(dot)
for i in range(0, 100):
    dot = [np.random.normal(center_b[0], chaos_b), np.random.normal(center_b[1], chaos_b)]
    dots.append(dot)

# Draw data
plt.scatter([x[0] for x in dots[0:100]], [x[1] for x in dots[0:100]], c='b', label="surrounded by" + str(center_a))
plt.scatter([x[0] for x in dots[100:200]], [x[1] for x in dots[100:200]], c='r', label="surrounded by" + str(center_b))
plt.title('DataForClassify')
plt.legend()

# Save data
## save pic
plt.savefig('../Data&Results/Classify/pic.png')
## save data as txt file
random.shuffle(dots)
'''change data to random list'''
filename = 'data.txt'
with open('../Data&Results/Classify/' + filename, 'w') as file:
    for dot in dots:
        file.write(str(dot[0]) + "," + str(dot[1]))
        file.write('\n')
plt.show()
