"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: ID3

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np
import math

def entropy(p):
    if p == 0 or p ==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)


class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,gain = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        gain : the gain of splitting the data according to 'x[self.feature] < self.theta ?'
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.gain = gain
        self.label = label


class DecisionTree(object):
    """ A decision tree for bianry classification.
        Training method: ID3
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """

        m, d= X.shape
        A = np.zeros((m-1,d))
        #xSort = np.column_stack(X)
        
        for j in range(d):
            x = X[:,j]
            np.unique(x)
            for i in range(m-1):
                A[i,j] = (x[i]+x[i+1])/2

        self.root = self.ID3(X,y,A,self.max_depth)



                


    def ID3(self,X, y, A, depth):
        """
        Gorw a decision tree with the ID3 recursive method

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        m , d = X.shape
        if not -1 in y :
            return Node(samples=m, label=1)
        if not 1 in y :
            return Node(samples=m, label=-1)

        if depth == 0:
            minus = np.count_nonzero(y == -1)
            plus = m - minus
            label = 1 if (plus > minus) else -1
            return Node(samples = m, label = label)
        gainMatrix = self.info_gain(X,y,A)
        maxGainInd = np.argmax(gainMatrix)
        r = maxGainInd / d
        j = maxGainInd % d
        gain = gainMatrix[r,j]
        theta  = A[r][j]
        feature = j
        Xs , Xb, ys, yb = [], [], [], []

        for i in range(m):
            if X[i,feature] > theta:
                Xb.append(X[i,:])
                yb.append(y[i])
            else:
                Xs.append(X[i,:])
                ys.append(y[i])

        if (len(Xb) == m) or (len(Xs) == m):
            plus = np.count_nonzero(y == 1)
            minus = m - plus
            l = (1 if minus < plus else -1)
            return Node(samples=m, label=l)

        Tl = self.ID3(np.array(Xs),np.array(ys),A,depth-1)
        Tr = self.ID3(np.array(Xb),np.array(yb),A,depth-1)

        return Node(left = Tl, right = Tr, samples = m, feature = feature, theta = theta,gain = gain)
        

    @staticmethod
    def info_gain(X, y, A):
        """
        Parameters
        ----------
        X, y : sample
        A : array of m*d real features, A[:,j] corresponds to thresholds over x_j

        Returns
        -------
        gain : m*d array containing the gain for each feature
        """

        m, d = X.shape
        cpy = entropy(float(np.count_nonzero(y == 1))/m)
        gainMatrix = np.zeros(A.shape)
        

        for i in range(d): #2

            x = X[:,i]
            for j in range(m-1):#499 
                theta = A[j,i]
                plus = np.count_nonzero(x>theta)
                yPlus = np.dot(x>theta,(0.5*(y+1)))
                #minus = np.count_nonzero(x<=theta)
                yMinus = np.dot(x<=theta,(0.5*(y+1)))
                
                minus = m - plus
                Pyx = [0.0] * 2
                if minus > 0:
                    Pyx[0] = float(yMinus)/minus
                if plus > 0:
                    Pyx[1] = float(yPlus)/plus
                gainMatrix[j,i] = cpy - (((float(plus)/m)*entropy(Pyx[1])) + ((float(minus)/m)*(entropy(Pyx[0]))))
        return gainMatrix
        



    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        yPred =[]
        for x in X:
            start = self.root

            while (start.label == None):
                if (x[start.feature] > start.theta):
                    start = start.right
                else:
                    start = start.left
            yPred.append(start.label)
        return np.array(yPred)


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        yHat = self.predict(X)
        loss = 0
        for i in range(len(y)):
            if yHat[i] != y[i]:
                loss += 1
        return float(loss) / len(y)
