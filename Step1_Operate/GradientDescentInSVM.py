# -*- coding: utf-8 -*-
"""
@Date : 2020/5/9 21:51
@File : GradientDescentInSVM.py
@author : SaltFish
@Tool : PyCharm

"""
import numpy as np
import matplotlib.pyplot as plt
import time


def Init():
    StepLength = 0.000001
    k = 0.3
    b = 25
    number = 10
    return StepLength, k, b, number


# Read the data
def ReadData():
    dots = []
    filename = "ClassifyTestDots.txt"
    file_path = '../Data&Results/Classify/' + filename
    with open(file_path) as file:
        for line in file:
            dot = line.rstrip().split(",")
            '''use rstrip here to delete the /n'''
            dot[0] = float(dot[0])
            dot[1] = float(dot[1])
            dot[2] = float(dot[2])
            dots.append(dot)
    return dots


def Operate(dots, k, b, StepLength):
    count = -1
    set_k = []
    set_b = []
    start = time.time()
    for i in range(0, 1000):
        d_k = 0
        d_b = 0
        for dot in dots:
            if k * dot[0] + b - dot[1] > 0:
                d_k += (((-b) * k + k * dot[1] + (b ** 2) * dot[0]) / pow(k ** 2 + b ** 2, 3 / 2))
                d_b += (((-k) * dot[0] * b + b * dot[1] + k ** 2) / pow(k ** 2 + b ** 2, 3 / 2))
            else:
                d_k += (((-b) * k + (-k) * dot[1] + (b ** 2) * dot[0]) / pow(k ** 2 + b ** 2, 3 / 2))
                d_b += (((-k) * dot[0] * b + (-b) * dot[1] + k ** 2) / pow(k ** 2 + b ** 2, 3 / 2))
        k_t = k + StepLength * d_k
        b_t = b + StepLength * d_b
        if abs(k - k_t) < 1e-5 and abs(b - b_t) < 1e-5:
            count = i
            break
        else:
            k = k_t
            b = b_t
            set_k.append(k)
            set_b.append(b)
    end = time.time()
    timeuse = round(end - start, 5)
    return set_k, set_b, count, timeuse


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
        plt.text(x[1], y[1], "No." + str(i))
        if i == int(set_i[-1]):
            plt.annotate("best match", xy=((x[1] + x[0]) / 2, (y[1] + y[0]) / 2), xytext=(x[1] / 2, y[1]),
                         arrowprops=dict(arrowstyle='->'))
    for dot in dots:
        if dot[2] == 1:
            plt.scatter(dot[0], dot[1], c='b')
        else:
            plt.scatter(dot[0], dot[1], c='r')
    plt.title('SVMResults')
    plt.text(min([x[0] for x in dots]), max([x[1] for x in dots]) / 2,
             'timeuse=' + str(timeuse) + "s\nround=" + str(count))
    plt.savefig('../Data&Results/Classify/SVMResult.png')
    plt.show()


# main function
def main():
    dots = ReadData()
    StepLength, k, b, number = Init()
    set_k, set_b, count, timeuse = Operate(dots, k, b, StepLength)
    draw(dots, set_k, set_b, number, count, timeuse)


if __name__ == '__main__':
    main()
