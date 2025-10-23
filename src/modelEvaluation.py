import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        self.predictions = {}
        self.metrics = {}
        
    def getPredictions(self, models, xTest):
        self.predictions = {
            "SVM": models['svm'].predict(xTest),
            "Random Forest": models['randomForest'].predict(xTest),
            "LightGBM": models['lightGbm'].predict(xTest),
            "MLP": (models['mlp'].predict(xTest) > 0.5).astype("int32").flatten()
        }
        return self.predictions
    
    def evaluateModel(self, yTest, yPred, modelName):
        print(f"--- Evaluation for {modelName} ---")
        targetNames = ['ADL', 'Fall']
        print(classification_report(yTest, yPred, target_names=targetNames))
        
        cm = confusion_matrix(yTest, yPred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=targetNames, yticklabels=targetNames)
        plt.title(f'Confusion Matrix for {modelName}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        return cm
    
    def evaluateAllModels(self, yTest):
        for name, yPred in self.predictions.items():
            self.evaluateModel(yTest, yPred, name)
    
    def getMetricsSummary(self, yTest):
        from sklearn.metrics import precision_recall_fscore_support
        
        summaryData = []
        
        for name, yPred in self.predictions.items():
            accuracy = accuracy_score(yTest, yPred)
            precision, recall, f1Score, _ = precision_recall_fscore_support(
                yTest, yPred, average=None
            )
            
            summaryData.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision (Fall)': precision[1],
                'Recall (Fall)': recall[1],
                'F1-Score (Fall)': f1Score[1]
            })
        
        self.metrics = pd.DataFrame(summaryData)
        return self.metrics
    
    def getBestModel(self):
        bestIdx = self.metrics['F1-Score (Fall)'].idxmax()
        return self.metrics.loc[bestIdx, 'Model']
