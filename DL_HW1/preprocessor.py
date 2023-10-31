from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import os

class energyPreprocessor():
    def __init__(self):
        self.sizeList = [5, 10, 15, 17]
        self.originalData = pd.read_csv('energy_efficiency_data.csv')
        self.encodedData = None
        self.normalizedX = None
        self.shuffledTrainX = self.shuffledTestX = self.shuffledTrainY = self.shuffledTestY = None
        self.trainX = self.testX = self.trainY = self.testY = None
        
        self.transformOHE()
        self.normalization()
        self.split()
        self.featureSelection()

    # transform category to OHE
    def transformOHE(self):
        encodedData = self.originalData
        encodedData = pd.get_dummies(encodedData, columns=['Glazing Area Distribution']) # for Glazing Area Distribution
        encodedData = pd.get_dummies(encodedData, columns=['Orientation']) # for Orientation
        self.encodedData = encodedData * 1 # transform True/False to 1/0

    # normalization
    def normalization(self):
        scaler = StandardScaler()
        self.normalizedX = pd.DataFrame(scaler.fit_transform(self.encodedData), columns=self.encodedData.columns)

    # split training, testing data
    def split(self):
        Y = self.encodedData["Heating Load"]
        X = self.normalizedX.drop(columns=['Heating Load'])
        self.shuffledTrainX, self.shuffledTestX, self.shuffledTrainY, self.shuffledTestY = train_test_split(X, Y, test_size=0.25, shuffle=True)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(X, Y, test_size=0.25, shuffle=False)

    # feature selection
    def featureSelection(self):
        for i in range(len(self.sizeList)):
            size = self.sizeList[i]
            selector = SelectKBest(score_func=f_regression, k=size)
            selector.fit_transform(self.shuffledTrainX, self.shuffledTrainY)
            selectedIndices = selector.get_support(indices=True)
            selectedNames = self.shuffledTrainX.columns[selectedIndices]
            
            # for shuffled
            shuffledFeatureSelectedTrainX = pd.DataFrame(self.shuffledTrainX, columns=selectedNames)
            shuffledFeatureSelectedTestX = pd.DataFrame(self.shuffledTestX, columns=selectedNames)
            
            # for unshuffled
            unshuffledFeatureSelectedTrainX = pd.DataFrame(self.trainX, columns=selectedNames)
            unshuffledFeatureSelectedTestX = pd.DataFrame(self.testX, columns=selectedNames)
            
            # dump csv file
            # for shuffled
            shuffledTrain = pd.concat([shuffledFeatureSelectedTrainX, self.shuffledTrainY], axis=1, ignore_index=False)
            shuffledTest = pd.concat([shuffledFeatureSelectedTestX, self.shuffledTestY], axis=1, ignore_index=False)
            shuffledTrain.to_csv(os.path.join('energy', 'shuffledTrain{}.csv'.format(size)), index=False)
            shuffledTest.to_csv(os.path.join('energy', 'shuffledTest{}.csv'.format(size)), index=False)

            # for unshuffled
            unshuffledTrain = pd.concat([unshuffledFeatureSelectedTrainX, self.trainY], axis=1, ignore_index=False)
            unshuffledTest = pd.concat([unshuffledFeatureSelectedTestX, self.testY], axis=1, ignore_index=False)
            unshuffledTrain.to_csv(os.path.join('energy', 'unshuffledTrain{}.csv'.format(size)), index=False)
            unshuffledTest.to_csv(os.path.join('energy', 'unshuffledTest{}.csv'.format(size)), index=False)
            
class ionospherePreprocessor():
    def __init__(self):
        self.originalData = pd.read_csv('ionosphere_data.csv', header=None).drop(columns=[1])
        self.trainX = self.testX = self.trainY = self.testY = None
        
        self.transformOHE()
        self.normalization()
        self.split()
        self.dumpFile()

    # transform ground truth to numerical values
    def transformOHE(self):
        category_mapping = {'b': 0, 'g': 1}
        self.originalData[34] = self.originalData[34].map(category_mapping)
        self.encodedData = self.originalData

    # normalization
    def normalization(self):
        scaler = StandardScaler()
        self.normalizedX = pd.DataFrame(scaler.fit_transform(self.encodedData), columns=self.encodedData.columns)

    # split training, testing data
    def split(self):
        Y = self.encodedData[34]
        X = self.normalizedX.drop(columns=[34])
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # dump csv file
    def dumpFile(self):
        train = pd.concat([self.trainX, self.trainY], axis=1, ignore_index=False)
        test = pd.concat([self.testX, self.testY], axis=1, ignore_index=False)
        train.to_csv(os.path.join('ionosphere', 'train.csv'), index=False)
        test.to_csv(os.path.join('ionosphere', 'test.csv'), index=False)