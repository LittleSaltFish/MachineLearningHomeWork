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
def Init():
    '''create dots surrounded by two centers'''
    ## set the center
    center_a = [25, 50]
    center_b = [50, 25]
    ## set the standard deviation
    chaos_a = 5
    chaos_b = 5
    size_a = 100
    size_b = 100
    return center_a, center_b, chaos_a, chaos_b, size_a, size_b


# Create data
def CreatDots(center_a, center_b, chaos_a, chaos_b, size_a, size_b):
    dots = []
    '''set of all dots'''
    for i in range(0, size_a):
        dot = [np.random.normal(center_a[0], chaos_a), np.random.normal(center_a[1], chaos_a), 1]
        dots.append(dot)
    for i in range(0, size_b):
        dot = [np.random.normal(center_b[0], chaos_b), np.random.normal(center_b[1], chaos_b), -1]
        dots.append(dot)
    return dots


# Draw data
def DrawAndSaveData(dots, size_a, size_b, name, dir, center_a, center_b, chaos_a, chaos_b, ran):
    plt.scatter([x[0] for x in dots[0:size_a]], [x[1] for x in dots[0:size_a]], c='b',
                label="surrounded by" + str(center_a) + ",chaos=" + str(chaos_a) + ",size=" + str(size_a))
    plt.scatter([x[0] for x in dots[size_a:size_a + size_b]], [x[1] for x in dots[size_a:size_a + size_b]], c='r',
                label="surrounded by" + str(center_b) + ",chaos=" + str(chaos_b) + ",size=" + str(size_b))
    plt.title(name)
    plt.legend()
    # Save data
    ## save pic
    plt.savefig(dir + name + ".png")
    ## save data as txt file
    if ran == True:
        random.shuffle(dots)
    '''change data to random list'''
    with open('../Data&Results/Classify/' + name + ".txt", 'w') as file:
        for dot in dots:
            file.write(str(dot[0]) + "," + str(dot[1]) + "," + str(dot[2]))
            file.write('\n')
    plt.show()


def main():
    center_a, center_b, chaos_a, chaos_b, size_a, size_b = Init()
    TrainDots = CreatDots(center_a, center_b, chaos_a, chaos_b, size_a, size_b)
    TestDots = CreatDots(center_a, center_b, chaos_a, chaos_b, size_a, size_b)
    DrawAndSaveData(TrainDots, size_a, size_b, "ClassifyTrainDots", '../Data&Results/Classify/', center_a, center_b,
                    chaos_a,
                    chaos_b, False)
    DrawAndSaveData(TestDots, size_a, size_b, "ClassifyTestDots", '../Data&Results/Classify/', center_a, center_b,
                    chaos_a,
                    chaos_b, True)


if __name__ == '__main__':
    main()
