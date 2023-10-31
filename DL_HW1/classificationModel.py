from preprocessor import ionospherePreprocessor
from model import ClassificationModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Optimizer: sgd
# Hyper parameter: # of hidden layers, # of hidden units, LR, # of epochs and mini-batch size

class config:
    miniBatchSize = 128
    visualizationDim = 3
    hiddenUnits = [33, 16, visualizationDim, 1]
    LR = 2e-4
    epochs = 10000

def main():
    # do data preprocessing
    dataProcessor = ionospherePreprocessor()
    
    # load shuffled training, testing data
    data = pd.read_csv(os.path.join('ionosphere', 'train.csv'))
    trainY = data['34'].values.reshape(-1, 1)
    trainX = data.drop(columns=['34']).values
    data = pd.read_csv(os.path.join('ionosphere', 'test.csv'))
    testY = data['34'].values.reshape(-1, 1)
    testX = data.drop(columns=['34']).values
    
    # training stage
    model = ClassificationModel(hiddenUnits=config.hiddenUnits, LR=config.LR)
    print("Training stage")
    CE_trainingHistory = []
    ER_trainingHistory = []
    ER_testingHistory = []
    for epoch in tqdm(range(config.epochs)):
        for j in range(0, len(trainX), config.miniBatchSize):
            start, end = j, j + config.miniBatchSize
            end = end if end < len(trainX) else len(trainX)
            batchX = trainX[start:end]
            batchY = trainY[start:end]
            _, _, predictY = model.forward(batchX)
            model.loss(batchY, predictY)
            model.backward()
        
        CE, trainingER = model.crossEntropy(), model.errorRate(trainY)
        CE_trainingHistory.append(CE)
        ER_trainingHistory.append(trainingER)
        model.reset()
        
        # testing stage
        _, _, predictY = model.forward(testX)
        model.loss(testY, predictY)
        testingER = model.errorRate(testY)
        ER_testingHistory.append(testingER)
        model.reset()
        
        if epoch % 1000 == 0:
            msg = "The {}th Epoch => Cross-entropy error(CE): {:.4f}, Training Error rate(ER): {:.4f}, Testing Error rate(ER): {:.4f}".format(epoch, CE, trainingER, testingER)
            tqdm.write(msg)
        
        if epoch == config.epochs / 10:
            _, l3, _ = model.forward(trainX)
            epoch1000_L3_latentFeature = l3
            
        if epoch == config.epochs / 10 * 9:
            _, l3, _ = model.forward(trainX)
            epoch9000_L3_latentFeature = l3

    # draw training CE history
    plt.figure(figsize = (16,8))
    plt.plot(np.array(CE_trainingHistory).reshape(-1,1),'-g', label = 'prediction',linewidth=2.0)
    plt.title("Cross Entropy Training Phase")
    plt.xlabel("#th Epoches ")
    plt.ylabel("Cross Entropy Error")
    plt.legend(loc='best')
    plt.savefig(os.path.join('ionosphere', 'CE_history.png'))

    # draw training & testing CE history
    plt.figure(figsize = (16,8))
    plt.plot(np.array(ER_trainingHistory).reshape(-1,1),'-r', label = 'training',linewidth=2.0)
    plt.plot(np.array(ER_testingHistory).reshape(-1,1),'-b', label = 'testing',linewidth=2.0)
    plt.title("Error Rate Training and Testing Phase")
    plt.xlabel("#th Epoches ")
    plt.ylabel("Error Rate")
    plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
    plt.savefig(os.path.join('ionosphere', 'ER_history.png'))
    
    # draw l3 latent feature
    if config.visualizationDim == 2:
        # for 2d
        good1000 = epoch1000_L3_latentFeature[(trainY == 1)[:, 0]]
        bad1000 = epoch1000_L3_latentFeature[(trainY == 0)[:, 0]]
        good9000 = epoch9000_L3_latentFeature[(trainY == 1)[:, 0]]
        bad9000 = epoch9000_L3_latentFeature[(trainY == 0)[:, 0]]
        plt.figure(figsize = (16, 8))
        plt.title("For the layer which has 2 neurons")
        plt.subplot(121)
        plt.plot(good1000[:, 0], good1000[:, 1], "go", label = "Good")
        plt.plot(bad1000[:, 0], bad1000[:, 1], "ro", label = "Bad")
        plt.title("2D feature 5000th epoch")
        plt.subplot(122)
        plt.plot(good9000[:, 0], good9000[:, 1], "go", label = "Good")
        plt.plot(bad9000[:, 0], bad9000[:, 1], "ro", label = "Bad")
        plt.title("2D feature 45000th epoch")
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.savefig(os.path.join('ionosphere', '2D-l3.png'))
        
    elif config.visualizationDim == 3:
        # draw 3d 
        good1000 = epoch1000_L3_latentFeature[(trainY == 1)[:, 0]]
        bad1000 = epoch1000_L3_latentFeature[(trainY == 0)[:, 0]]
        good9000 = epoch9000_L3_latentFeature[(trainY == 1)[:, 0]]
        bad9000 = epoch9000_L3_latentFeature[(trainY == 0)[:, 0]]
        ax = plt.subplot(121, projection = '3d')
        ax.scatter(good1000[:, 0], good1000[:, 1], good1000[:, 2], marker = 'o', color = "green", label = "Good", alpha = 1.0)
        ax.scatter(bad1000[:, 0], bad1000[:, 1], bad1000[:, 2], marker = 'o', color = "red", label = "Bad", alpha = 1.0)
        plt.title("3D feature 5000th epoch")
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        ax = plt.subplot(122, projection = '3d')
        ax.scatter(good9000[:, 0], good9000[:, 1], good9000[:, 2], marker = 'o', color = "green", alpha = 1.0, label = "Good")
        ax.scatter(bad9000[:, 0], bad9000[:, 1], bad9000[:, 2], marker = 'o', color = "red", alpha = 1.0, label = "Bad")
        plt.title("3D feature 45000th epoch")
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.savefig(os.path.join('ionosphere', '3D-l3.png'))
        
if __name__ == '__main__':
    main()