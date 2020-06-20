import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split  # 制作数据集和测试集
import graphviz
import pydotplus
import os


def getdata():
    '''抽取训练集'''
    DataSet = np.genfromtxt('DataSet3.0alpha.txt',
                            delimiter = ',',
                            encoding  = 'UTF-8')[1:]
    # 过滤第一行标签，设置UTF-8避免报错
    size = abs(np.random.normal(0.5, 0.15)) % 1
    #随机生成测试集大小，正态分布避免过少的数据，用abs和%1避免越界
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        DataSet[:, 1:3], DataSet[:, -1], test_size=size)
    #只要训练集
    print('Xtrain='+str(Xtrain))
    print('Ytrain='+str(Ytrain))
    print('size='+str(size))
    return Xtrain, Ytrain,DataSet


def MakeTree(DataSet, Xtrain, Ytrain, number):
    '''建立决策树'''
    clf        = tree.DecisionTreeClassifier(criterion="entropy")  # 载入决策树分类模型
    clf.fit(Xtrain, Ytrain)  # 决策树拟合，得到模型
    TrainScore = clf.score(Xtrain, Ytrain)  # 预测的准确度
    TestScore  = clf.score(DataSet[:,1:3],DataSet[:,-1])   #不能要编号
    result     = clf.predict(DataSet[:, 1:3])
    print('result='+str(result))
    print('TrainScore='+str(TrainScore))
    print('TestScore='+str(TestScore))
    
    '''画图'''
    FeatureName = ['roh', 'sugar']
    ClassName   = ['good', 'bad']
    dot_data    = tree.export_graphviz(
        clf, out_file=None, feature_names=FeatureName, class_names=ClassName, filled=True, rounded=True)
    os.environ["PATH"] += os.pathsep+'D:\\软件（拷贝）\\GraphViz\\bin'
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("TreeStructure\\"+str(number)+"_out.png")
    return result


def main():
    TreeNumber = 10     #定义基桩数目
    results    = []     #输出集合
    for i in range(0, TreeNumber): 
        Xtrain, Ytrain, DataSet = getdata()
        #有放回的随机选择测试集
        results.append(MakeTree(DataSet,Xtrain, Ytrain,i))
        #加入单个预测结果
    results = np.array(results)
    #转化为numpy数组以使用size()
    print('results='+str(results))
    ans=[]      #bagging结果
    for i in range(0,np.size(results,1)):
        ans.append(1 if sum(results[:,i])>0 else -1)
        #求和，相当于投票。如果大于0则好瓜，小于0则坏瓜
    print('ans='+str(ans))

if __name__ == '__main__':
    main()
