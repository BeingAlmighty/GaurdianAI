import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class ModelTrainer:
    def __init__(self):
        self.svmModel = None
        self.rfModel = None
        self.lgbmModel = None
        self.mlpModel = None
        self.history = None
        
    def trainSvm(self, xTrain, yTrain):
        self.svmModel = SVC(
            probability=True, 
            random_state=42,
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced'
        )
        self.svmModel.fit(xTrain, yTrain)
        return self.svmModel
    
    def trainRandomForest(self, xTrain, yTrain):
        self.rfModel = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1
        )
        self.rfModel.fit(xTrain, yTrain)
        return self.rfModel
    
    def trainLightGbm(self, xTrain, yTrain):
        classWeights = compute_class_weight('balanced', classes=np.unique(yTrain), y=yTrain)
        sampleWeights = np.array([classWeights[int(i)] for i in yTrain])
        
        self.lgbmModel = lgb.LGBMClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=15,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        self.lgbmModel.fit(xTrain, yTrain, sample_weight=sampleWeights)
        return self.lgbmModel
    
    def buildMlpModel(self, inputShape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(inputShape,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def trainMlp(self, xTrain, yTrain, xVal, yVal):
        classWeights = compute_class_weight('balanced', classes=np.unique(yTrain), y=yTrain)
        classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
        
        self.mlpModel = self.buildMlpModel(xTrain.shape[1])
        
        earlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduceLr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        self.history = self.mlpModel.fit(
            xTrain, yTrain,
            validation_data=(xVal, yVal),
            epochs=100,
            batch_size=64,
            class_weight=classWeightDict,
            callbacks=[earlyStop, reduceLr],
            verbose=1
        )
        
        return self.mlpModel, self.history
    
    def trainAllModels(self, xTrain, yTrain, xVal, yVal):
        print("Training SVM with balanced class weights...")
        self.trainSvm(xTrain, yTrain)
        
        print("\nTraining Random Forest with balanced class weights...")
        self.trainRandomForest(xTrain, yTrain)
        
        print("\nTraining LightGBM with balanced class weights...")
        self.trainLightGbm(xTrain, yTrain)
        
        print("\nTraining Enhanced MLP with regularization...")
        self.trainMlp(xTrain, yTrain, xVal, yVal)
        
        return {
            'svm': self.svmModel,
            'randomForest': self.rfModel,
            'lightGbm': self.lgbmModel,
            'mlp': self.mlpModel
        }
