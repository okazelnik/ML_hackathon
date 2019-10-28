"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex6.

Author:
Date: April, 2016

"""
import numpy as np
import ex6_tools 
from adaboost import AdaBoost
from matplotlib import pyplot
import matplotlib.pyplot as plt
import decision_tree 
import nearest_neighbors


def Q3(): # AdaBoost
    # TODO - implement this function
    path = 'C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\SynData'
    dataxVal = np.loadtxt(path + '\\X_val.txt')
    datayVal = np.loadtxt(path + '\\y_val.txt')
    data_x = np.loadtxt(path + '\\X_train.txt')
    data_y = np.loadtxt(path + '\\y_train.txt')
    dataTestX = np.loadtxt(path + '\\X_test.txt')
    dataTestY = np.loadtxt(path + '\\y_test.txt')

##    dataxVal = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\X_val.txt')
##    datayVal = np.loadtxt('C:\\Users\Omri\\Desktop\\Omri\\studies\\IML\\y_val.txt')
##    data_x = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\X_train.txt')
##    data_y = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\y_train.txt')
##    dataTestX = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\X_test.txt')
##    dataTestY = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\y_test.txt')


    wl = ex6_tools.DecisionStump
    T =range(0,205,5)
    T[0] = 1
    T1 = [1,5,10,50,100,200]
    
    errorTrain = []
    errorValidtion = []
    ab = []
    i=0
    for t in T:
        ab.append(AdaBoost(wl,t))
        ab[i].train(data_x,data_y)
        errorTrain.append(ab[i].error(data_x,data_y))
        errorValidtion.append(ab[i].error(dataxVal,datayVal))
        
        if t in T1:
            ex6_tools.decision_boundaries(ab[i],data_x,data_y,"T : "+ str(t),None)
        i+=1

    plt.plot(np.array(T),np.array(errorTrain))
    plt.plot(np.array(T),np.array(errorValidtion))
    plt.show()

    #part6
    minValError = np.argmin(np.array(errorValidtion))

    tHat = T[minValError]
    err = ab[minValError].error(dataTestX,dataTestY)
    title = "errorValidtion: "+str(errorValidtion[minValError])+" T: "+str(tHat)+" Error: "+str(err)
    ex6_tools.decision_boundaries(ab[minValError],data_x,data_y, title,None)
    return 0

def Q4(): # decision trees
    # TODO - implement this function
    

    path = 'C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\SynData'
    dataxVal = np.loadtxt(path + '\\X_val.txt')
    datayVal = np.loadtxt(path + '\\y_val.txt')
    data_x = np.loadtxt(path + '\\X_train.txt')
    data_y = np.loadtxt(path + '\\y_train.txt')
    dataTestX = np.loadtxt(path + '\\X_test.txt')
    dataTestY = np.loadtxt(path + '\\y_test.txt')
    
    dtClassifier = []
    maxDepth = 12
    errorTrain =[]
    errorValidtion = []
    for i in range(maxDepth):
        dtClassifier.append(decision_tree.DecisionTree(i + 1))
        dtClassifier[i].train(data_x,data_y)
        ex6_tools.decision_boundaries(dtClassifier[i],data_x,data_y, i+1 ,None)
        errorTrain.append(dtClassifier[i].error(data_x,data_y))
        errorValidtion.append(dtClassifier[i].error(dataxVal,datayVal))
        
    plt.plot(np.array(range(1,13)),np.array(errorTrain))
    plt.plot(np.array(range(1,13)),np.array(errorValidtion))
    plt.show()

    minValidError  = np.argmin(np.array(errorValidtion))
    print dtClassifier[minValidError].error(dataTestX,dataTestY)
    feature_names = np.array(['x1','x2'])
    class_names = np.array(['0','1'])
    #ex6_tools.view_dtree(dtClassifier[minValidError],feature_names,class_names,"dtree.pdf")
    return

def Q5(): # kNN
    
    path = 'C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\SynData'
    dataxVal = np.loadtxt(path + '\\X_val.txt')
    datayVal = np.loadtxt(path + '\\y_val.txt')
    data_x = np.loadtxt(path + '\\X_train.txt')
    data_y = np.loadtxt(path + '\\y_train.txt')
    dataTestX = np.loadtxt(path + '\\X_test.txt')
    dataTestY = np.loadtxt(path + '\\y_test.txt')

    K = [1,3,10,100,200,500]
    nnClassifier =[]
    errorTrain =[]
    errorValidtion = []
    for k in K: 
        nn = nearest_neighbors.kNN(k)
        nn.train(data_x,data_y)
        nnClassifier.append(nn)
        ex6_tools.decision_boundaries(nn,dataxVal,datayVal, k ,None)
        errorTrain.append(nn.error(data_x,data_y))
        errorValidtion.append(nn.error(dataxVal,datayVal))

    plt.plot(np.log(K),np.array(errorTrain))
    plt.plot(np.log(K),np.array(errorValidtion))
    plt.show()

    minValidError  = np.argmin(np.array(errorValidtion))
    print nnClassifier[minValidError].error(dataTestX,dataTestY)

    return

def Q6(): # Republican or Democrat?
    
    y = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\CongressData\\parties.txt') #y
    X = np.loadtxt('C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\CongressData\\votes.txt') #x
    m, d = X.shape
    
    trainX ,trainy = [], []
    valX , valy = [], []
    testX , testy = [], []

    permutation = np.random.permutation(m)
    for i in range(m):
        if i <= (m*0.5):
            trainX.append(X[permutation[i]]);
            trainy.append(y[permutation[i]]);
        elif (i>m*0.5) and (i<=m*0.9):
            valX.append(X[permutation[i]]);
            valy.append(y[permutation[i]]);
        else :
            testX.append(X[permutation[i]]);
            testy.append(y[permutation[i]]);
    trainX =  np.array(trainX)
    trainy = np.array(trainy)
    valX = np.array(valX)
    valy = np.array(valy)
    testX = np.array(testX)
    testy = np.array(testy)


#adaBoost
    wl = ex6_tools.DecisionStump
    T = range(0,100,10)
    T[0] = 1
    adc = []
    errorTrainAdc = []
    errorValAdc = []
    for i in range(len(T)):
        adc.append(AdaBoost(wl,T[i]))
        adc[i].train(trainX,trainy)
        errorTrainAdc.append(adc[i].error(trainX,trainy))
        errorValAdc.append(adc[i].error(valX,valy))
    err = errorTrainAdc.index(min(errorTrainAdc))
    print "AdaBoost T=%d,\t val error: %f \t test error %f" %(T[err],errorValAdc[err], adc[err].error(testX,testy))

#decision tree
    dtc =[]
    depth =range(1,12)
    errorTrainDtc = []
    errorValADtc = []
    for i in range(len(depth)):
        dtc.append(decision_tree.DecisionTree(depth[i]))
        dtc[i].train(trainX,trainy)
        errorTrainDtc.append(dtc[i].error(trainX,trainy))
        errorValADtc.append(dtc[i].error(valX,valy))
    err = errorTrainDtc.index(min(errorTrainDtc))
    print "decision tree depth=%d,\t val error: %f \t test error %f" %(depth[err],errorValADtc[err], dtc[err].error(testX,testy))

#kNN
    knnc = []
    K = [3,10,100,150]
    errorTrainKnn = []
    errorValKnn = []
    for i in range(len(K)):
        knnc.append(nearest_neighbors.kNN(K[i]))
        knnc[i].train(trainX,trainy)
        errorTrainKnn.append(knnc[i].error(trainX,trainy))
        errorValKnn.append(knnc[i].error(valX,valy))
    err = errorTrainKnn.index(min(errorTrainKnn))
    print "kNN K=%d,\t val error: %f \t test error %f" %(K[err],errorValKnn[err], knnc[err].error(testX,testy))
    
    return 0


if __name__ == '__main__':
    #Q3()
    #Q4()
    #Q5()
    Q6()
