# -*- coding: utf-8 -*-
"""
@Date : 2020/5/10 11:31
@File : SVM-3D.py
@author : SaltFish
@Tool : PyCharm

"""
import mayavi.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


def Init():
    k = 1
    b = 0
    size = 50
    return k, b, size


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


def GetDistent(dots, k, b, size):
    set_k = np.linspace(k - size, k + size, 2 * size)
    set_b = np.linspace(b - size, b + size, 2 * size)
    z = zeros(shape=(len(set_k), len(set_b)))
    k, b = np.meshgrid(set_k, set_b)
    for dot in dots:
        z += abs(k * dot[0] + b - dot[1]) / sqrt(k * k + b * b)
    return k, b, z


def DrawAndSaveDis_mlab(k, b, z):
    pl = mlab.surf(k, b, z, warp_scale="auto")
    mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    mlab.outline(pl)
    mlab.show()


def DrawAndSaveDis_matplot(k, b, z):
    fig = figure()
    ax = Axes3D(fig)
    ax.plot_surface(k, b, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('k Label')
    ax.set_ylabel('b Label')
    ax.set_zlabel('SumDis Label')
    ax.view_init(50, 90)
    savefig("0001")
    plt.show()


def main():
    k, b, size = Init()
    dots = ReadData()
    k, b, z = GetDistent(dots, k, b, size)
    DrawAndSaveDis_matplot(k, b, z)


if __name__ == '__main__':
    main()
