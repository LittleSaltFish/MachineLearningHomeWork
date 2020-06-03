#
# MultipleClassification-SVM.py
# @author Saltfish
# @description 
# @created 2020-05-20T15:25:39.796Z+08:00
# @last-modified 2020-05-23 T 23:18:21.172 Z +08:00
# @Email saltfish@whut.edu.cn
# @license     MIT License
#    
#    Copyright (c) 2020 Saltfish
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#

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
from sklearn.model_selection import train_test_split


def LoadData():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0) 

    return x_train, x_test, y_train, y_test


def Draw(x, y, name):
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='k', label='setosa')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='b', label='versicolour')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], c='r', label='virginica')
    plt.xlabel('length')
    plt.ylabel('width')
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig('./' + name + '.png')
    plt.show()


def Classify(x_train, x_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_ans = clf.predict(x_test)
    return y_ans


def DrawErr(x_test, y_test, y_ans):
    dic_ErrDot = {}
    for i in range(len(y_test)):
        key = (str(y_test[i]) + '->' + str(y_ans[i]))
        if y_ans[i] != y_test[i]:
            if key not in dic_ErrDot.keys():
                dic_ErrDot[key] = []
            dic_ErrDot[key].append([x_test[i][0], x_test[i][1]])
    plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='k', edgecolor='g', label='right:setosa', alpha=0.7)
    plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='b', edgecolor='g', label='right:versicolour',
                alpha=0.7)
    plt.scatter(x_test[y_test == 2, 0], x_test[y_test == 2, 1], c='r', edgecolor='g', label='right:virginica',
                alpha=0.7)
    count = 0
    color = ['crimson', 'aqua']
    for (name, ErrDots) in dic_ErrDot.items():
        plt.scatter([ErrDot[0] for ErrDot in ErrDots], [ErrDot[1] for ErrDot in ErrDots], c=color[count], marker='x',
                    s=150, label='wrong:' + name)
        count += 1
    plt.legend(loc='best')
    plt.xlabel('length')
    plt.ylabel('width')
    plt.title('Iris Error Dots')
    plt.savefig('./ErrorDots.png')
    plt.show()


def main():
    x_train, x_test, y_train, y_test = LoadData()
    Draw(x_train, y_train, 'IrisData-Train')
    Draw(x_test, y_test, 'IrisData-Test')
    y_ans = Classify(x_train, x_test, y_train, y_test)
    Draw(x_test, y_ans, 'IrisData-Answer')
    DrawErr(x_test, y_test, y_ans)
    print('done')


if __name__ == '__main__':
    main()
