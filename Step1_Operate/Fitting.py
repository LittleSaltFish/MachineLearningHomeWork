# -*- coding: utf-8 -*-
"""
@Date : 2020/5/8 15:53
@File : Fitting].py
@author : SaltFish
@Tool : PyCharm

"""
import matplotlib.pyplot as plt
import numpy as np
import time


# Parameter settings
def Init():
    StepLength_k = 0.000001
    StepLength_b = 0.000001
    k = 0
    b = 0
    number = 50
    return StepLength_k, StepLength_b, k, b, number


def Fitting(dots, StepLength_k, StepLength_b, k, b):
    start = time.time()
    set_k = []
    set_b = []
    count = 0
    for j in range(0, 1000):
        sum_k = 0
        sum_b = 0
        for dot in dots:
            sum_k += (b + k * dot[0] - dot[1]) * dot[0]
            sum_b += b + k * dot[0] - dot[1]
        k = k - StepLength_k * sum_k / len(dots)
        b = b - StepLength_b * sum_b / len(dots)
        set_k.append(k)
        set_b.append(b)
        count = j
        if abs(StepLength_k * sum_k / len(dots)) < 1e-5 and abs(StepLength_b * sum_b / len(dots)) < 1e-5:
            break
        end = time.time()
        timeuse = round(end - start,5)
    return set_k, set_b, count, timeuse


# Read the data
def ReadData():
    dots = []
    filename = "data.txt"
    file_path = '../Data&Results/Fitting/' + filename
    with open(file_path) as file:
        for line in file:
            dot = line.rstrip().split(",")
            '''use rstrip here to delete the /n'''
            dot[0] = float(dot[0])
            dot[1] = float(dot[1])
            dots.append(dot)
    return dots


# Draw and save
def draw(dots, set_k, set_b, number, count, timeuse):
    colors = []
    num_color = np.linspace(1068683, 2003199, num=number, endpoint=True)
    for num in num_color:
        num = int(num)
        colors.append("#" + str(hex(num))[2:])
    set_i = np.linspace(0, len(set_k) - 1, num=number, endpoint=True)
    for i, color in zip(set_i, colors):
        i = int(i)
        x = [min(x[0] for x in dots), max(x[0] for x in dots)]
        y = [set_k[i] * x[0] + set_b[i], set_k[i] * x[1] + set_b[i]]
        plt.plot(x, y, color=color)
        if i==int(set_i[-1]):
            plt.annotate("best match",xy=(x[1]/2,y[1]/2),xytext=(x[1]/2,y[1]),arrowprops=dict(arrowstyle='->'))
    plt.scatter([x[0] for x in dots], [x[1] for x in dots], c='b')
    plt.title('FittingResults')
    plt.text(0,200,'timeuse='+str(timeuse)+"s\nround="+str(count))
    plt.savefig('../Data&Results/Fitting/FittingResult.png')
    plt.show()


# main function
def main():
    dots = ReadData()
    StepLength_k, StepLength_b, k, b, number = Init()
    set_k, set_b, count, timeuse = Fitting(dots, StepLength_k, StepLength_b, k, b)
    draw(dots, set_k, set_b, number, count, timeuse)


if __name__ == '__main__':
    main()
