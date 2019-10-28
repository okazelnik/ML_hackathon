import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

xpath = 'C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\X_poly.npy'
ypath = 'C:\\Users\\Omri\\Desktop\\Omri\\studies\\IML\\exs\\Y_poly.npy'


def vandermonde(dim,x):
    matrixSize = x.size
    dimX = dim+1
    van = np.zeros((dimX,matrixSize))
    for i in range(dimX):
        for j in range(matrixSize):
            van[i][j] = pow(x[j],i)
    return van

def calcTraining(dim,x,y):
    matrix = vandermonde(dim,x)
    matrixTranspose = np.transpose(matrix)
    matrixInverse = np.linalg.pinv(matrixTranspose)
    w =  np.dot(matrixInverse,y)
    return w

def calcValidation(w,x,y,dim):
    xSize = x.size
    matrix = vandermonde(dim,x)
    return (np.sum(np.power(np.dot(np.transpose(matrix),w) - y,2))/(xSize*1.0))

def calcError(wStar,x,y,dim):
    xSize = x.size
    matrix = vandermonde(dim,x)
    return (np.sum(np.power(np.dot(np.transpose(matrix),wStar) - y,2))/(xSize*1.0))



def k_fold(data_x,data_y):
    K = 5
    Sx = np.split(data_x,K)
    Sy = np.split(data_y,K)
    errorArray = np.array([])
    for i in range(1,16):
        errorK = np.array([])     
        for j in range(K):
            xTmp = np.array([])
            yTmp = np.array([])
            for n in range(K):
                if n != j:
                    xTmp = np.append(xTmp,Sx[n])
                    yTmp = np.append(yTmp,Sy[n])
            h = calcTraining(i,xTmp,yTmp)
            errorK = np.append(errorK,[calcError(h,Sx[j],Sy[j],i)])
        errorArray = np.append(errorArray,[np.sum(errorK)/(K*1.0)])
    
    estar = np.argmin(errorArray)
    return calcTraining(estar+1,data_x,data_y)
  

def ex4(X_file, Y_file):
    data_x = np.load(X_file)
    data_y = np.load(Y_file)
    
    [training_x ,validation_x,test_x] = np.split(data_x,3)
    [training_y ,validation_y,test_y] = np.split(data_y,3)

    #print calcTraining(15,training_x,training_y)
    H ={}
    validationH = np.zeros(15)
    trainError = np.array([])
    validationError = np.array([])

    for i in range(1,16):
        H[i] = calcTraining(i,training_x,training_y)
        validationH[i-1] = calcValidation(H[i],validation_x,validation_y,i)
        trainError = np.append(trainError,[calcError(H[i],training_x,training_y,i)])
        validationError = np.append(validationError,[calcError(H[i],validation_x,validation_y,i)])

    print validationError
    X = range(1,16)
    plt.title('validation : blue   traning :red')
    plt.plot(X,validationError ,'b-',label='validation')
    plt.plot(X,trainError,'r-',label = 'traning')
    plt.show()
    # BEST PLUS ERROR
    hstar = np.argmin(validationH)+1
    test = calcError(H[hstar],test_x,test_y,hstar)

    print k_fold(data_x[0:200],data_y[0:200])

    return 0



ex4(xpath,ypath)
