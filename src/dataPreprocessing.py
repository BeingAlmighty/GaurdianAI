import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, filePath):
        self.filePath = filePath
        self.scaler = StandardScaler()
        self.df = None
        self.xTrain = None
        self.xVal = None
        self.xTest = None
        self.yTrain = None
        self.yVal = None
        self.yTest = None
        
    def loadData(self):
        self.df = pd.read_csv(self.filePath)
        return self.df
    
    def getClassDistribution(self):
        return self.df['label'].value_counts(normalize=True)
    
    def checkMissingValues(self):
        return self.df.isnull().sum().sum()
    
    def scaleFeatures(self):
        x = self.df.drop('label', axis=1)
        y = self.df['label']
        
        xScaled = self.scaler.fit_transform(x)
        
        return xScaled, y
    
    def splitData(self):
        xScaled, y = self.scaleFeatures()
        
        xTrain, xTemp, yTrain, yTemp = train_test_split(
            xScaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        xVal, xTest, yVal, yTest = train_test_split(
            xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp
        )
        
        self.xTrain = xTrain
        self.xVal = xVal
        self.xTest = xTest
        self.yTrain = yTrain
        self.yVal = yVal
        self.yTest = yTest
        
        return xTrain, xVal, xTest, yTrain, yVal, yTest
