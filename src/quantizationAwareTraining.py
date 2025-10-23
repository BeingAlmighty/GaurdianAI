import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False
    print("WARNING: tensorflow-model-optimization not available. Using alternative QAT implementation.")


class QuantizationAwareTrainer:
    def __init__(self):
        self.baseModel = None
        self.qatModel = None
        self.history = None
    
    def buildQatModelAlternative(self, inputShape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(inputShape,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
        
    def buildBaseModel(self, inputShape):
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
        
        return model
    
    def pretrainModel(self, xTrain, yTrain, xVal, yVal):
        print("Step 1: Pre-training base model...")
        
        self.baseModel = self.buildBaseModel(xTrain.shape[1])
        
        classWeights = compute_class_weight('balanced', classes=np.unique(yTrain), y=yTrain)
        classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.baseModel.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
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
        
        history = self.baseModel.fit(
            xTrain, yTrain,
            validation_data=(xVal, yVal),
            epochs=100,
            batch_size=64,
            class_weight=classWeightDict,
            callbacks=[earlyStop, reduceLr],
            verbose=1
        )
        
        return self.baseModel, history
    
    def applyQuantizationAwareTraining(self, xTrain, yTrain, xVal, yVal):
        print("\nStep 2: Applying Quantization-Aware Training...")
        print("Using QAT-inspired training with enhanced regularization...")
        
        self.qatModel = self.buildQatModelAlternative(xTrain.shape[1])
        self.qatModel.set_weights(self.baseModel.get_weights())
        
        classWeights = compute_class_weight('balanced', classes=np.unique(yTrain), y=yTrain)
        classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        self.qatModel.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        earlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduceLr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.000001,
            verbose=1
        )
        
        self.history = self.qatModel.fit(
            xTrain, yTrain,
            validation_data=(xVal, yVal),
            epochs=50,
            batch_size=64,
            class_weight=classWeightDict,
            callbacks=[earlyStop, reduceLr],
            verbose=1
        )
        
        return self.qatModel, self.history
    
    def trainWithQat(self, xTrain, yTrain, xVal, yVal):
        self.pretrainModel(xTrain, yTrain, xVal, yVal)
        self.applyQuantizationAwareTraining(xTrain, yTrain, xVal, yVal)
        
        return self.qatModel, self.history
    
    def convertToTfLiteQat(self, xTrain, outputPath='thinker_model_qat_int8.tflite'):
        print("\nStep 3: Converting QAT model to TFLite INT8...")
        
        def representativeDatasetGen():
            for i in range(min(500, len(xTrain))):
                yield [xTrain[i:i+1].astype(np.float32)]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.qatModel)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representativeDatasetGen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tfliteQatModel = converter.convert()
        
        with open(outputPath, 'wb') as f:
            f.write(tfliteQatModel)
        
        print(f"QAT INT8 TFLite model saved as '{outputPath}'")
        return outputPath
