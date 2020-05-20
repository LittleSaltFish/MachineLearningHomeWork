# -*- coding: utf-8 -*-
"""
@Date : 2020/5/8 15:53
@File : Classify.py
@author : SaltFish
@Tool : PyCharm

"""

from sklearn import svm
import matplotlib.pyplot as plt


# Read the data
def ReadData(filename):
    dots = []
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


def TrainAndTest(trainset, testset):
    set_results = []
    clf = svm.SVC(C=1)  # class
    clf.fit([[x[0], x[1]] for x in trainset], [x[2] for x in trainset])  # training the svc model
    results = clf.predict([[x[0], x[1]] for x in testset])  # predict the target of testing samples
    for dot, result in zip(testset, results):
        set_results.append([dot[0], dot[1], result])
    return set_results, clf.support_vectors_, clf.support_, clf.n_support_


def DrawAndSaveData(set_results, support, name, dir):
    draw_s1 = []
    draw_s2 = []
    draw_n1 = []
    draw_n2 = []
    for i in range(0, len(set_results)):
        if i in support and set_results[i][2] == 1:
            draw_s1.append([set_results[i][0], set_results[i][1], set_results[i][2]])
        elif i in support and set_results[i][2] == -1:
            draw_s2.append([set_results[i][0], set_results[i][1], set_results[i][2]])
        elif not (i in support) and set_results[i][2] == 1:
            draw_n1.append([set_results[i][0], set_results[i][1], set_results[i][2]])
        elif not (i in support) and set_results[i][2] == -1:
            draw_n2.append([set_results[i][0], set_results[i][1], set_results[i][2]])
    plt.scatter([dot[0] for dot in draw_s1], [dot[1] for dot in draw_s1], c='b', marker='o',
                label="support dots surround by dot1")
    plt.scatter([dot[0] for dot in draw_s2], [dot[1] for dot in draw_s2], c='r', marker='o',
                label="support dots surround by dot2")
    plt.scatter([dot[0] for dot in draw_n1], [dot[1] for dot in draw_n1], c='b', marker='x',
                label="normal dots surround by dot1")
    plt.scatter([dot[0] for dot in draw_n2], [dot[1] for dot in draw_n2], c='r', marker='x',
                label="normal dots surround by dot2")
    plt.title(name)
    plt.legend()
    plt.savefig(dir + name + ".png")
    plt.show()


def main():
    trainset = ReadData("ClassifyTrainDots.txt")
    testset = ReadData("ClassifyTestDots.txt")
    set_results, support_vectors, support, n_support = TrainAndTest(trainset, testset)
    DrawAndSaveData(set_results, support, "SVM results", '../Data&Results/Classify/')
    print(set_results)


if __name__ == '__main__':
    main()
