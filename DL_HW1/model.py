import numpy as np
import math

EPS = 1e-15
size = 5

class RegressionModel():
    def __init__(self, inputSize=size, hiddenUnits=10, LR=2e-4):
        # initializaion
        self.errorRecord = np.array([])
        self.first = True
        self.LR = LR
        
        # warning: use standard (np.random.rand)normal distribution instead of np.random.rand
        np.random.seed(0)
        self.w1 = np.random.randn(inputSize, hiddenUnits)
        self.b1 = np.zeros((1, hiddenUnits))
        self.w2 = np.random.randn(hiddenUnits, hiddenUnits)
        self.b2 = np.zeros((1, hiddenUnits))
        self.w3 = np.random.randn(hiddenUnits, 1)
        self.b3 = np.zeros((1, 1))
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
        
    def forward(self, x1):
        self.x1 = x1
        self.a1 = x1.dot(self.w1) + self.b1
        self.x2 = self.sigmoid(self.a1)
        self.a2 = self.x2.dot(self.w2) + self.b2
        self.x3 = self.sigmoid(self.a2)
        self.a3 = self.x3.dot(self.w3) + self.b3
        return self.a3
    
    """ sum-of-squares error """
    def loss(self, groundTruth, predict):
        # calculate error
        self.error = predict - groundTruth
        
        # record error
        if self.first:
            self.errorRecord = self.error
            self.first = False
        else:
            self.errorRecord = np.concatenate((self.errorRecord, self.error), axis=0)
    
    def reset(self):
        self.first = True
    
    def sumSquareError(self):
        return np.square(self.errorRecord).sum()
    
    def rootMeanSquare(self):
        return math.sqrt(np.square(self.errorRecord).mean())
        
    def backward(self):
        dloss_da3 = 2 * self.error
        dloss_dx3 = dloss_da3.dot(self.w3.T)
        dloss_dw3 = self.x3.T.dot(dloss_da3)
        dloss_db3 = np.sum(dloss_da3, axis=0, keepdims=True)
        
        dx3_da2 = self.x3 * (1 - self.x3)
        dloss_da2 = dloss_dx3 * dx3_da2
        dloss_dx2 = dloss_da2.dot(self.w2.T)
        dloss_dw2 = self.x2.T.dot(dloss_da2)
        dloss_db2 = np.sum(dloss_da2, axis=0, keepdims=True)
        
        dx2_da1 = self.x2 * (1 - self.x2)
        dloss_da1 = dloss_dx2 * dx2_da1
        dloss_dx1 = dloss_da1.dot(self.w1.T)
        dloss_dw1 = self.x1.T.dot(dloss_da1)
        dloss_db1 = np.sum(dloss_da1, axis=0, keepdims=True)
        
        self.w3 -= dloss_dw3 * self.LR
        self.b3 -= dloss_db3 * self.LR
        self.w2 -= dloss_dw2 * self.LR
        self.b2 -= dloss_db2 * self.LR
        self.w1 -= dloss_dw1 * self.LR
        self.b1 -= dloss_db1 * self.LR

class ClassificationModel():
    def __init__(self, hiddenUnits=[33, 20, 20, 1], LR=2e-5):
        # initializaion
        self.first = True
        self.LR = LR
        
        # warning: use standard (np.random.rand)normal distribution instead of np.random.rand
        np.random.seed(0)
        self.w1 = np.random.randn(hiddenUnits[0], hiddenUnits[1])
        self.b1 = np.zeros((1, hiddenUnits[1]))
        self.w2 = np.random.randn(hiddenUnits[1], hiddenUnits[2])
        self.b2 = np.zeros((1, hiddenUnits[2]))
        self.w3 = np.random.randn(hiddenUnits[2], hiddenUnits[3])
        self.b3 = np.zeros((1, hiddenUnits[3]))
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
        
    def softmax(self, X):
        return np.exp(X) / np.exp(X).sum()
        
    def forward(self, x1):
        self.x1 = x1
        self.a1 = x1.dot(self.w1) + self.b1
        self.x2 = self.sigmoid(self.a1)
        self.a2 = self.x2.dot(self.w2) + self.b2
        self.x3 = self.sigmoid(self.a2)
        self.a3 = self.x3.dot(self.w3) + self.b3
        self.x4 = self.sigmoid(self.a3)
        #! warning: maybe we should use softmax
        return self.a1, self.a2, self.x4
    
    def loss(self, groundTruth, predict):
        # calculate error
        self.Terror = predict - groundTruth
        # prevent result to be nan after exp operation 
        boundedPredict = predict
        boundedPredict[boundedPredict < EPS] = EPS
        boundedPredict[boundedPredict > 1 - EPS] = 1 - EPS
        self.error = (groundTruth * np.log(boundedPredict) + (1 - groundTruth) * np.log(1 - boundedPredict))
        
        # record error
        if self.first:
            self.errorRecord = self.error
            self.predictRecord = predict
            self.first = False
        else:
            self.errorRecord = np.concatenate((self.errorRecord, self.error), axis=0)
            self.predictRecord = np.concatenate((self.predictRecord, predict), axis=0)
    
    def reset(self):
        self.first = True
    
    def crossEntropy(self):
        m = len(self.errorRecord)
        return -1 / m * (self.errorRecord).sum()
    
    def errorRate(self, groundTruth):
        predictLabel = self.predictRecord
        predictLabel[predictLabel > 0.5] = 1
        predictLabel[predictLabel < 0.5] = 0
        errorAmount = (predictLabel != groundTruth).sum()
        totalAmount = len(groundTruth)
        return errorAmount / totalAmount
        
    def backward(self):
        #! modify dloss_dx4 to dervation of CE
        dloss_dx4 = 2 * self.Terror
        dx4_da3 = self.x4 * (1 - self.x4)
        dloss_da3 = dloss_dx4 * dx4_da3
        dloss_dx3 = dloss_da3.dot(self.w3.T)
        dloss_dw3 = self.x3.T.dot(dloss_da3)
        dloss_db3 = np.sum(dloss_da3, axis=0, keepdims=True)
        
        dx3_da2 = self.x3 * (1 - self.x3)
        dloss_da2 = dloss_dx3 * dx3_da2
        dloss_dx2 = dloss_da2.dot(self.w2.T)
        dloss_dw2 = self.x2.T.dot(dloss_da2)
        dloss_db2 = np.sum(dloss_da2, axis=0, keepdims=True)
        
        dx2_da1 = self.x2 * (1 - self.x2)
        dloss_da1 = dloss_dx2 * dx2_da1
        dloss_dx1 = dloss_da1.dot(self.w1.T)
        dloss_dw1 = self.x1.T.dot(dloss_da1)
        dloss_db1 = np.sum(dloss_da1, axis=0, keepdims=True)
        
        self.w3 -= dloss_dw3 * self.LR
        self.b3 -= dloss_db3 * self.LR
        self.w2 -= dloss_dw2 * self.LR
        self.b2 -= dloss_db2 * self.LR
        self.w1 -= dloss_dw1 * self.LR
        self.b1 -= dloss_db1 * self.LR