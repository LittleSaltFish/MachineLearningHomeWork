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
from pylab import *
import mayavi.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D


def Init():
    StepLength = 0.000001
    k = 1
    b = 0
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
        if abs(k - k_t) < 1e-6 and abs(b - b_t) < 1e-6:
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
    eg_k = []
    eg_b = []
    num_color = np.linspace(1068683, 2003199, num=number, endpoint=True)
    for num in num_color:
        num = int(num)
        colors.append("#" + str(hex(num))[2:])
    set_i = np.linspace(0, len(set_k) - 1, num=number, endpoint=True)
    for i, color in zip(set_i, colors):
        i = int(i)
        eg_k.append(set_k[i])
        eg_b.append(set_b[i])
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
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.text(min([x[0] for x in dots]), max([x[1] for x in dots]) / 2,
             'timeuse=' + str(timeuse) + "s\nround=" + str(count))
    plt.savefig('../Data&Results/Classify/SVMResult.png')
    plt.show()
    return eg_k, eg_b


def GetSampleDistent(dots, eg_k, eg_b):
    eg_z = []
    for k, b in zip(eg_k, eg_b):
        z = 0
        for dot in dots:
            z += abs(k * dot[0] + b - dot[1]) / sqrt(k * k + b * b)
        eg_z.append(z)
    return eg_z


def GetDistentMaterix(dots, k, b, size):
    m_k = np.linspace(k - size, k + size, 2 * size)
    m_b = np.linspace(b - size, b + size, 2 * size)
    m_z = zeros(shape=(len(m_k), len(m_b)))
    m_k, m_b = np.meshgrid(m_k, m_b)
    for dot in dots:
        m_z += abs(m_k * dot[0] + m_b - dot[1]) / sqrt(m_k * m_k + m_b * m_b)
    return m_k, m_b, m_z


def DrawAndSaveDis_matplot(m_k, m_b, m_z, eg_k, eg_b, eg_z):
    fig = figure()
    ax = Axes3D(fig)
    ax.plot_surface(m_k, m_b, m_z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.plot3D(eg_k, eg_b, eg_z, color='gray')
    ax.scatter(eg_k, eg_b, eg_z, color='k')
    ax.set_xlabel('k Label')
    ax.set_ylabel('b Label')
    ax.set_zlabel('SumDis Label')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    ax.view_init(50, 60)
    savefig('../Data&Results/Classify/3DResult.png')
    plt.show()


# main function
def main():
    dots = ReadData()
    StepLength, k, b, number = Init()
    set_k, set_b, count, timeuse = Operate(dots, k, b, StepLength)
    eg_k, eg_b = draw(dots, set_k, set_b, number, count, timeuse)
    eg_z = GetSampleDistent(dots, eg_k, eg_b)
    m_k, m_b, m_z = GetDistentMaterix(dots, k, b, 20)
    DrawAndSaveDis_matplot(m_k, m_b, m_z, eg_k, eg_b, eg_z)
    print('done')

if __name__ == '__main__':
    main()
