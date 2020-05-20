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
def Init():
    '''create dots surrounded by line"kx+b" '''
    k = 0.5
    b = 10
    chaos_x = 5
    chaos_y = 50
    x = 0
    return k, b, chaos_x, chaos_y, x


def CreateDots(x, k, b, chaos_x, chaos_y):
    # Create data
    dots = []
    for i in range(0, 100):
        x += abs(np.random.normal(0, chaos_x))
        y = x * k + b + np.random.normal(0, chaos_y)
        dots.append([x, y])
    return dots


def DrawAndSaveData(dots, dir, name):
    # Draw data
    plt.scatter([x[0] for x in dots], [x[1] for x in dots], c='b')
    plt.title('DataForFitting')

    # Save data
    ## save pic
    plt.savefig(dir + name + ".png")
    ## save data as txt file
    random.shuffle(dots)
    '''change data to random list'''
    with open(dir + name + ".txt", 'w') as file:
        for dot in dots:
            file.write(str(dot[0]) + "," + str(dot[1]))
            file.write('\n')
    plt.show()


def main():
    k, b, chaos_x, chaos_y, x = Init()
    dots = CreateDots(x, k, b, chaos_x, chaos_y)
    DrawAndSaveData(dots, '../Data&Results/Fitting/', 'FittingData')


if __name__ == '__main__':
    main()
