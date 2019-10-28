"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np
import math

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m = X.shape[0]
        D = np.empty(m)
        D.fill(1/(m*1.0))

        hs = 0
        for t in range(self.T):
            self.h[t] = self.WL(D,X,y)
            eps = 0

            for j in range(m):   
                if (y[j]*(1.0) != self.h[t].predict(X)[j]):
                    eps+=D[j]

            self.w[t] =0.5*math.log((1/eps)-1)

            Dtmp = []
            for j in range(m):
                Dtmp.append(D[j]*math.exp(-1*self.w[t]*y[j]*(self.h[t].predict(X)[j])))

            Dtmp = np.array(Dtmp)   
            s = np.sum(Dtmp)

            for i in range(m):
                D[i] = (D[i]*math.exp(-1*self.w[t]*y[i]*self.h[t].predict(X)[i]))/s

            
        





    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        hs = 0;
        for t in range(self.T):
            hs+=self.w[t]*self.h[t].predict(X)
        return np.sign(hs)



    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        matches=[]
        pred = self.predict(X)
        for i in range(len(X)):
            if y[i]!= pred[i]:
                matches.append(1)
            else:
                matches.append(0)
        matches = np.array(matches)
        return np.sum(matches)*(1/(len(matches)*1.0))

