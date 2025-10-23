import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")
        
    def plotClassDistribution(self, df):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='label', data=df)
        plt.title('Class Distribution (0: ADL, 1: Fall)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()
        
    def plotConfusionMatrix(self, cm, modelName, colorMap='Blues'):
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap=colorMap, 
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        plt.title(f'Confusion Matrix for {modelName}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def plotTrainingHistory(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
