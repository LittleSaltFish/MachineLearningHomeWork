# -*- coding: utf-8 -*-
"""
@Date : 2020/5/20 15:25
@File : multi.py
@author : SaltFish
@Tool : PyCharm

"""

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def LoadAndDraw():
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_sepal = x[:, :2]
    plt.scatter(x_sepal[:, 0], x_sepal[:, 1], c=y, cmap=plt.cm.rainbow)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Dots')
    plt.savefig('../Data&Results/Iris-Data.png')
    plt.show()

def main():
    LoadAndDraw()
    print('done')

if __name__ == '__main__':
    main()