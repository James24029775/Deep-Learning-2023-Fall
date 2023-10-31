from preprocessor import energyPreprocessor
from model import RegressionModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Optimizer: sgd
# Hyper parameter: # of hidden layers, # of hidden units, LR, # of epochs and mini-batch size

class config:
    size = 5
    epochs = 10000
    miniBatchSize = 128
    shuffledTrainFile = 'shuffledTrain{}.csv'.format(size)
    shuffledTestFile = 'shuffledTest{}.csv'.format(size)
    unshuffledTrainFile = 'unshuffledTrain{}.csv'.format(size)
    unshuffledTestFile = 'unshuffledTest{}.csv'.format(size)
    
def main():
    # do data preprocessing
    dataProcessor = energyPreprocessor()
    
    # load shuffled training, testing data
    data = pd.read_csv(os.path.join('energy', config.shuffledTrainFile))
    trainY = data["Heating Load"].values.reshape(-1, 1)
    trainX = data.drop(columns=['Heating Load']).values
    data = pd.read_csv(os.path.join('energy', config.shuffledTestFile))
    testY = data["Heating Load"].values.reshape(-1, 1)
    testX = data.drop(columns=['Heating Load']).values
    
    # training stage
    model = RegressionModel()
    print("Training stage")
    SSE_trainingHistory = []
    RMS_trainingHistory = []
    for epoch in tqdm(range(config.epochs)):
        for j in range(0, len(trainX), config.miniBatchSize):
            start, end = j, j + config.miniBatchSize
            end = end if end < len(trainX) else len(trainX)
            batchX = trainX[start:end]
            batchY = trainY[start:end]
            predictY = model.forward(batchX)
            model.loss(batchY, predictY)
            model.backward()
            
        SSE, RMS = model.sumSquareError(), model.rootMeanSquare()
        SSE_trainingHistory.append(SSE)
        RMS_trainingHistory.append(RMS)
        if epoch % 1000 == 0:
            msg = "The {}th Epoch => Sum-of-squares error(SSE): {:.4f}, Root-mean-square error(RMS): {:.4f}".format(epoch, SSE, RMS)
            tqdm.write(msg)
    
        model.reset()
    
    # inference stage
    print("Testing stage")
    predictY = model.forward(testX)
    model.loss(testY, predictY)
    SSE, RMS = model.sumSquareError(), model.rootMeanSquare()
    print("Sum-of-squares error(SSE): {:.4f}, Root-mean-square error(RMS): {:.4f}".format(SSE, RMS))
    model.reset()
    
    # draw SSE training history
    plt.figure(figsize = (16,8))
    plt.plot(np.array(SSE_trainingHistory).reshape(-1,1),'-g', label = 'Training Curve Error',linewidth=2.0)
    plt.title("Training Curve")
    plt.xlabel("Number of Epochs ")
    plt.ylabel("Sum-Of-Squares Error")
    plt.legend(loc = 'best')
    plt.savefig(os.path.join('energy', 'TrainingCurveError.png'))
    
    # load unshuffled training, testing data
    data = pd.read_csv(os.path.join('energy', config.unshuffledTrainFile))
    trainY = data["Heating Load"].values.reshape(-1, 1)
    trainX = data.drop(columns=['Heating Load']).values
    data = pd.read_csv(os.path.join('energy', config.unshuffledTestFile))
    testY = data["Heating Load"].values.reshape(-1, 1)
    testX = data.drop(columns=['Heating Load']).values

    # draw unshuffled train history
    predictY = model.forward(trainX)
    plt.figure(figsize = (16,8))
    plt.plot(predictY, 'r-', label = 'prediction',linewidth = 2)
    plt.plot(trainY, 'g-', label=' label',linewidth = 2)
    plt.title("Prediction for Training Data with No Shuffle")
    plt.xlabel("#th Case")
    plt.ylabel("Heating Load")
    plt.legend(loc='best')
    plt.savefig(os.path.join('energy', 'UnshuffledTrain.png'))
    
    # draw unshuffled test history
    predictY = model.forward(testX)
    plt.figure(figsize = (16,8))
    plt.plot(predictY, 'r-', label = 'prediction',linewidth = 2)
    plt.plot(testY, 'g-', label=' label',linewidth = 2)
    plt.title("Prediction for Testing Data with No Shuffle")
    plt.xlabel("#th Case ")
    plt.ylabel("Heating Load")
    plt.legend(loc='best')
    plt.savefig(os.path.join('energy', 'UnshuffledTest.png'))

if __name__ == '__main__':
    main()