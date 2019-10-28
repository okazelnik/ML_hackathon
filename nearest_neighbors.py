"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the k nearest neighbors classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np
class kNN(object):

    def __init__(self, k):
        # TODO - implement this method
        self.k = k
        self.X = None
        self.y = None
        return

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        self.X = X
        self.y = y



    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        m, d= X.shape
        
        yHat = [0]*m
        #distX = np.zeros((m,m))
        for i in range(m):
            positiveLabel,negativeLabel = 0 , 0
            x = X[i,]
            distX = np.linalg.norm(self.X - x, axis=1)
            sortDist = np.argsort(distX)[:self.k]

            for j in range(self.k):
                if (self.y[sortDist[j]]) == 1:

                    positiveLabel += 1
                else:
                    negativeLabel += 1
            yHat[i] = (1 if positiveLabel > negativeLabel else -1)
        return np.array(yHat)

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        m, d = X.shape
        yHat = self.predict(X)
        loss = 0
        for i in range(m):
            if yHat[i] != y[i]:
                loss += 1
        return float(loss) / m


